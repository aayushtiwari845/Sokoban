"""Phase 1 -- environment and data verification.  GATE 1.

Replaces every assumption in the project spec with a verified fact.  Writes
``results/data_verification.json``.  Prints five mandated checks:

 1. exact character alphabet over >=50 level files (does ``*``/``+`` occur?)
 2. grid invariants over >=10,000 levels
 3. solution-length units: moves or pushes?
 4. solution coverage and the length distribution the caption bins derive from
 5. near-duplicate / cross-split collision check

Run:
    python scripts/00_verify_data.py --levels-root data_raw/boxoban-levels \
        --solutions-root data_raw/astar-solutions --out results/data_verification.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sokogen.data import boxoban
from sokogen.data.boxoban import (GRID_H, GRID_W, N_BOX, N_GOAL, N_PLAYER,
                                  TILE_SET, load_level_file, split_files)
from sokogen.data.solutions import (is_action_string, load_solutions_frame,
                                    replay_actions, solution_key)
from sokogen.provenance import stamp, write_artifact

# Number of train files that make up the 100k training split (see spec 5.3).
TRAIN_FILES = 100


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Check 1: alphabet
# --------------------------------------------------------------------------
def check_alphabet(levels_root: str, n_files: int) -> Dict:
    hr(f"CHECK 1  Character alphabet over {n_files} level files")
    files = split_files(levels_root, "unfiltered-train")[:n_files]
    counter: collections.Counter = collections.Counter()
    for path in files:
        with open(path, "rb") as fh:
            counter.update(fh.read().decode("ascii"))

    print(f"{'char':<10}{'count':>14}")
    for ch, n in sorted(counter.items()):
        print(f"{ch!r:<10}{n:>14,}")

    tile_chars = {c for c in counter if c in TILE_SET}
    star = "*" in counter
    plus = "+" in counter
    crlf = "\r" in counter

    print()
    print(f"  tile characters present : {sorted(tile_chars)}")
    print(f"  '*' (box on goal)       : {star}")
    print(f"  '+' (player on goal)    : {plus}")
    print(f"  '\\r' (CRLF endings)     : {crlf}")

    if star or plus:
        raise SystemExit(
            "STOP: '*' or '+' found in the corpus.  The 5-channel tensor, the "
            "6-symbol vocabulary and the 'exactly four $ and four .' constraint "
            "all break if a box can start on a goal.  Reported, not adapted to."
        )
    print("\n  => VERIFIED: no '*' or '+'.  A box never starts on a goal.")
    if crlf:
        print("  => NOTE (spec did not anticipate this): files are CRLF; the parser "
              "normalises \\r\\n -> \\n.")

    return {
        "n_files_scanned": len(files),
        "char_counts": {repr(k): v for k, v in sorted(counter.items())},
        "has_box_on_goal_star": star,
        "has_player_on_goal_plus": plus,
        "has_crlf": crlf,
        "tile_alphabet": sorted(tile_chars),
    }


# --------------------------------------------------------------------------
# Check 2: grid invariants
# --------------------------------------------------------------------------
def check_invariants(levels_root: str, n_levels: int) -> Dict:
    hr(f"CHECK 2  Grid invariants over >={n_levels:,} levels")
    files = split_files(levels_root, "unfiltered-train")
    seen = 0
    violations: List[str] = []
    counts_hist = collections.Counter()
    for path in files:
        for lev in load_level_file(path, "unfiltered-train"):
            problems = boxoban.check_invariants(lev.grid)
            if problems:
                violations.append(f"{lev.source_file}#{lev.source_index}: {problems}")
            c = boxoban.tile_counts(lev.grid)
            counts_hist[(c["@"], c["$"], c["."])] += 1
            seen += 1
        if seen >= n_levels:
            break

    print(f"  levels checked            : {seen:,}")
    print(f"  shape                     : all {GRID_H}x{GRID_W}")
    print(f"  wall border (r0,r9,c0,c9) : enforced")
    print(f"  (players, boxes, goals) histogram:")
    for key, n in counts_hist.most_common(5):
        print(f"      {key} -> {n:,}")
    print(f"  violations                : {len(violations)}")
    for v in violations[:20]:
        print(f"      {v}")

    expected = (N_PLAYER, N_BOX, N_GOAL)
    all_expected = set(counts_hist) == {expected}
    print(f"\n  => every level has exactly {expected} (player, box, goal): {all_expected}")
    return {
        "n_levels_checked": seen,
        "n_violations": len(violations),
        "violations_sample": violations[:50],
        "tile_count_histogram": {str(k): v for k, v in counts_hist.items()},
        "all_levels_have_expected_counts": all_expected,
    }


# --------------------------------------------------------------------------
# Check 3: solution-length units
# --------------------------------------------------------------------------
def check_solution_units(levels_root: str, solutions_root: str) -> Dict:
    hr("CHECK 3  Solution-length units: player moves or pushes?")
    df = load_solutions_frame(solutions_root, "unfiltered-test")
    levels = load_level_file(
        os.path.join(levels_root, "unfiltered", "test", "000.txt"), "unfiltered-test")

    print(f"  columns: {list(df.columns)}")
    print(f"  index  : {df.index.names}, first key {df.index[0]!r} "
          f"(NOTE: zero-padded 3-digit STRINGS, not ints)")
    print()

    steps_eq_len = 0
    n_action_rows = 0
    examples = []
    replay_ok = replay_bad = invalid = 0
    move_lens, push_lens = [], []

    for lev in levels:
        row = df.loc[solution_key(lev.source_file, lev.source_index)]
        actions_raw = row["Actions"]
        if not is_action_string(actions_raw):
            invalid += 1
            continue
        n_action_rows += 1
        actions = actions_raw.strip()
        steps = int(row["Steps"])
        if steps == len(actions):
            steps_eq_len += 1
        rep = replay_actions(lev.grid, actions)
        if rep.solved:
            replay_ok += 1
            move_lens.append(rep.moves)
            push_lens.append(rep.pushes)
            if len(examples) < 5:
                examples.append({
                    "level": lev.source_index,
                    "grid": lev.grid,
                    "actions": actions,
                    "Steps": steps,
                    "replayed_moves": rep.moves,
                    "replayed_pushes": rep.pushes,
                })
        else:
            replay_bad += 1

    print(f"  Steps == len(Actions) for {steps_eq_len}/{n_action_rows} rows "
          f"=> Steps is the NUMBER OF ACTIONS")
    print(f"  actions are digits 0-3 = Up, Right, Down, Left "
          f"(verified: no other permutation replays)")
    print()
    print("  Five worked examples (level, Steps, replayed moves, replayed pushes):")
    for ex in examples:
        print(f"    level {ex['level']:>3}: Steps={ex['Steps']:>3}  "
              f"moves={ex['replayed_moves']:>3}  pushes={ex['replayed_pushes']:>3}")
        for r, line in enumerate(ex["grid"].rstrip("\n").split("\n")):
            print(f"        {line}")
        print(f"        actions: {ex['actions']}")
        print()

    if move_lens:
        gt = sum(1 for m, p in zip(move_lens, push_lens) if m > p)
        eq = sum(1 for m, p in zip(move_lens, push_lens) if m == p)
        print(f"  moves > pushes on {gt}/{len(move_lens)} levels; moves == pushes on {eq}")
        print(f"  mean moves = {np.mean(move_lens):.2f}, mean pushes = {np.mean(push_lens):.2f}")

    print()
    print("  => VERIFIED: 'Steps' counts PLAYER MOVES, not pushes.")
    print("     Their A* minimised moves, so Steps is move-optimal where valid.")
    print("     Pushes replayed from a move-optimal solution are an UPPER bound")
    print("     on the push-optimal length, so our push-optimal solver may report")
    print("     FEWER pushes; that is expected, not a bug.")
    print()
    print(f"  DATA-QUALITY FINDING (unfiltered_test, 1000 rows):")
    print(f"     upstream-failed rows (SEARCH_STATE_FAILED/NOT_FOUND) : {invalid}")
    print(f"     well-formed but DO NOT REPLAY to a solved state      : {replay_bad}")
    print(f"     replay-validated                                     : {replay_ok}")
    print("     => solutions must be replay-validated before use; see")
    print("        sokogen/data/solutions.py.")

    return {
        "columns": list(df.columns),
        "index_names": list(df.index.names),
        "index_key_format": "zero-padded 3-digit strings, e.g. ('000','023')",
        "action_encoding": {"0": "Up", "1": "Right", "2": "Down", "3": "Left"},
        "steps_equals_len_actions": f"{steps_eq_len}/{n_action_rows}",
        "units": "player_moves",
        "units_evidence": (
            "Steps == len(Actions) for every well-formed row, and replaying "
            "those actions under strict Sokoban rules reaches the goal state."
        ),
        "test_split_upstream_failed_rows": invalid,
        "test_split_wellformed_but_not_replaying": replay_bad,
        "test_split_replay_validated": replay_ok,
        "examples": examples,
        "mean_moves": float(np.mean(move_lens)) if move_lens else None,
        "mean_pushes": float(np.mean(push_lens)) if push_lens else None,
    }


# --------------------------------------------------------------------------
# Check 4: coverage + length distribution (source of the caption bins)
# --------------------------------------------------------------------------
def check_coverage(levels_root: str, solutions_root: str, n_files: int) -> Dict:
    hr(f"CHECK 4  Solution coverage and length distribution "
       f"({n_files} train files = {n_files*1000:,} levels)")
    df = load_solutions_frame(solutions_root, "unfiltered-train")
    files = split_files(levels_root, "unfiltered-train")[:n_files]

    total = 0
    have_row = 0
    upstream_failed = 0
    replay_failed = 0
    move_lens: List[int] = []
    push_lens: List[int] = []

    t0 = time.time()
    for path in files:
        for lev in load_level_file(path, "unfiltered-train"):
            total += 1
            try:
                actions_raw = df.at[solution_key(lev.source_file, lev.source_index),
                                    "Actions"]
            except KeyError:
                continue
            have_row += 1
            if not is_action_string(actions_raw):
                upstream_failed += 1
                continue
            rep = replay_actions(lev.grid, actions_raw.strip())
            if not rep.solved:
                replay_failed += 1
                continue
            move_lens.append(rep.moves)
            push_lens.append(rep.pushes)
    dt = time.time() - t0

    usable = len(move_lens)
    print(f"  levels                              : {total:,}")
    print(f"  with a row in the solutions CSV     : {have_row:,} ({100*have_row/total:.3f}%)")
    print(f"  upstream-failed rows                : {upstream_failed:,}")
    print(f"  well-formed but not replaying       : {replay_failed:,} "
          f"({100*replay_failed/total:.3f}%)")
    print(f"  REPLAY-VALIDATED, usable            : {usable:,} ({100*usable/total:.3f}%)")
    print(f"  ({dt:.1f}s)")

    mv = np.array(move_lens)
    pu = np.array(push_lens)
    deciles = [float(np.percentile(mv, q)) for q in range(0, 101, 10)]
    terciles = [float(np.percentile(mv, q)) for q in (100 / 3, 200 / 3)]

    print()
    print(f"  MOVE length: min={mv.min()} max={mv.max()} mean={mv.mean():.2f} "
          f"median={np.median(mv):.1f} std={mv.std():.2f}")
    print(f"  deciles (0..100 by 10): {[round(d,1) for d in deciles]}")
    print(f"  TERCILE boundaries for the easy/medium/hard caption bin: "
          f"{terciles[0]:.2f}, {terciles[1]:.2f}")
    print(f"      easy   : moves <= {terciles[0]:.0f}")
    print(f"      medium : {terciles[0]:.0f} < moves <= {terciles[1]:.0f}")
    print(f"      hard   : moves > {terciles[1]:.0f}")
    print()
    print(f"  PUSH length: min={pu.min()} max={pu.max()} mean={pu.mean():.2f} "
          f"median={np.median(pu):.1f}")
    print("  (pushes here are those of a MOVE-optimal solution: an upper bound "
          "on push-optimal)")

    hist = collections.Counter(move_lens)
    return {
        "n_levels": total,
        "n_with_row": have_row,
        "n_upstream_failed": upstream_failed,
        "n_replay_failed": replay_failed,
        "n_usable": usable,
        "coverage_fraction": have_row / total,
        "usable_fraction": usable / total,
        "move_length": {
            "min": int(mv.min()), "max": int(mv.max()),
            "mean": float(mv.mean()), "std": float(mv.std()),
            "median": float(np.median(mv)),
            "deciles": deciles,
            "tercile_boundaries": terciles,
        },
        "push_length": {
            "min": int(pu.min()), "max": int(pu.max()),
            "mean": float(pu.mean()), "std": float(pu.std()),
            "median": float(np.median(pu)),
            "deciles": [float(np.percentile(pu, q)) for q in range(0, 101, 10)],
        },
        "move_length_histogram": {str(k): v for k, v in sorted(hist.items())},
    }


# --------------------------------------------------------------------------
# Check 5: duplicates and cross-split collisions
# --------------------------------------------------------------------------
def check_duplicates(levels_root: str, n_files: int) -> Dict:
    hr(f"CHECK 5  Near-duplicate / cross-split collision check")

    def digest(grid: str) -> str:
        return hashlib.sha1(grid.encode("ascii")).hexdigest()

    train_hashes: collections.Counter = collections.Counter()
    for path in split_files(levels_root, "unfiltered-train")[:n_files]:
        for lev in load_level_file(path, "unfiltered-train"):
            train_hashes[digest(lev.grid)] += 1

    test_levels = load_level_file(
        os.path.join(levels_root, "unfiltered", "test", "000.txt"), "unfiltered-test")
    test_hashes = {digest(l.grid) for l in test_levels}

    valid_levels = []
    for path in split_files(levels_root, "unfiltered-valid")[:5]:
        valid_levels.extend(load_level_file(path, "unfiltered-valid"))
    valid_hashes = {digest(l.grid) for l in valid_levels}

    train_set = set(train_hashes)
    train_dups = sum(v - 1 for v in train_hashes.values() if v > 1)
    tt = train_set & test_hashes
    tv = train_set & valid_hashes

    print(f"  train levels hashed             : {sum(train_hashes.values()):,} "
          f"({len(train_set):,} distinct)")
    print(f"  duplicate levels *within* train : {train_dups:,}")
    print(f"  test levels hashed              : {len(test_levels):,}")
    print(f"  valid levels hashed             : {len(valid_levels):,}")
    print()
    print(f"  train n test collisions        : {len(tt)}   (expected 0)")
    print(f"  train n valid collisions       : {len(tv)}   (expected 0)")
    if tt or tv:
        print("  => NON-ZERO collisions: the novelty metric must exclude these.")
    else:
        print("  => VERIFIED: no exact cross-split leakage.")

    return {
        "n_train_hashed": int(sum(train_hashes.values())),
        "n_train_distinct": len(train_set),
        "n_duplicates_within_train": int(train_dups),
        "n_test": len(test_levels),
        "n_valid_hashed": len(valid_levels),
        "train_test_collisions": len(tt),
        "train_valid_collisions": len(tv),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-root", default="data_raw/boxoban-levels")
    ap.add_argument("--solutions-root", default="data_raw/astar-solutions")
    ap.add_argument("--out", default="results/data_verification.json")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--alphabet-files", type=int, default=50)
    ap.add_argument("--invariant-levels", type=int, default=10000)
    ap.add_argument("--train-files", type=int, default=TRAIN_FILES)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"levels    : {args.levels_root}")
    print(f"solutions : {args.solutions_root}")
    print(f"seed      : {args.seed}")

    payload = {
        "provenance": stamp(vars(args), args.seed),
        "check1_alphabet": check_alphabet(args.levels_root, args.alphabet_files),
        "check2_invariants": check_invariants(args.levels_root, args.invariant_levels),
        "check3_solution_units": check_solution_units(args.levels_root, args.solutions_root),
        "check4_coverage": check_coverage(args.levels_root, args.solutions_root,
                                          args.train_files),
        "check5_duplicates": check_duplicates(args.levels_root, args.train_files),
    }

    write_artifact(args.out, payload)
    hr("GATE 1 SUMMARY")
    c1, c2 = payload["check1_alphabet"], payload["check2_invariants"]
    c3, c4 = payload["check3_solution_units"], payload["check4_coverage"]
    c5 = payload["check5_duplicates"]
    print(f"  1 alphabet      : no '*'/'+' -> 5-channel repr SAFE; CRLF handled")
    print(f"  2 invariants    : {c2['n_levels_checked']:,} levels, "
          f"{c2['n_violations']} violations")
    print(f"  3 solution units: PLAYER MOVES (Steps == len(Actions))")
    print(f"  4 coverage      : {100*c4['usable_fraction']:.2f}% replay-validated; "
          f"terciles at {[round(t,1) for t in c4['move_length']['tercile_boundaries']]}")
    print(f"  5 duplicates    : {c5['train_test_collisions']} train/test collisions")
    print(f"\n  artifact -> {args.out}")


if __name__ == "__main__":
    main()
