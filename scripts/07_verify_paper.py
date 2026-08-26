"""Cross-check every numeric claim in the paper against the JSON artifacts.

Operating rule 5 says tables are generated from artifacts and never typed. The
tables obey that, but the paper's *prose* quotes numbers by hand, and a
transcription error there is exactly the failure the rule exists to prevent.

This script re-derives each quoted value from ``results/*.json`` and reports any
mismatch. It is a test of the manuscript, not of the code.

    python scripts/07_verify_paper.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Checker:
    def __init__(self, tol: float = 0.05):
        self.tol = tol
        self.rows: List[Tuple[str, Any, Any, bool]] = []

    @staticmethod
    def _decimals(x: Any) -> int:
        s = repr(float(x))
        return len(s.split(".")[1].rstrip("0")) if "." in s else 0

    def check(self, label: str, quoted: Any, actual: Any,
              tol: float | None = None) -> None:
        """A quoted value matches if the artifact rounds to it.

        The paper quotes rounded numbers, so the right test is
        ``round(actual, d) == quoted`` at the precision the paper used, not an
        absolute tolerance -- an absolute tolerance fails exactly on the
        half-way cases that rounding handles correctly (21.75 -> 21.8).
        A tolerance is still accepted for values quoted to a coarser precision
        than they were computed.
        """
        if isinstance(quoted, bool) or isinstance(actual, bool):
            ok = quoted == actual
        elif isinstance(quoted, (int, float)) and isinstance(actual, (int, float)):
            d = self._decimals(quoted)
            ok = round(float(actual), d) == float(quoted)
            if not ok and tol is not None:
                ok = abs(float(quoted) - float(actual)) <= tol
            elif not ok and tol is None:
                ok = abs(float(quoted) - float(actual)) <= self.tol
        else:
            ok = quoted == actual
        self.rows.append((label, quoted, actual, ok))

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows) + 2
        bad = 0
        for label, quoted, actual, ok in self.rows:
            q = f"{quoted:.4g}" if isinstance(quoted, float) else str(quoted)
            a = f"{actual:.4g}" if isinstance(actual, float) else str(actual)
            mark = "ok " if ok else "MISMATCH"
            if not ok:
                bad += 1
            print(f"  [{mark}] {label:<{width}} paper={q:<12} artifact={a}")
        print()
        print(f"  {len(self.rows) - bad}/{len(self.rows)} claims verified")
        return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    def load(name):
        with open(os.path.join(args.results, name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    ev = load("evaluation.json")
    sv = load("solver_validation.json")
    dv = load("data_verification.json")
    seeds = load("seed_variance.json")
    cond = ev["conditions"]
    c = Checker()

    def valid(n):
        return cond[n]["structural_validity"]["pct"]

    def solv(n):
        return cond[n]["solvable_given_valid"]["pct"]

    def drawn(n):
        return cond[n]["samples_drawn"]

    def tiles(n, tile, key):
        return cond[n]["tile_counts_raw_draws"][tile][key]

    print("=" * 78)
    print("Section V: Solver")
    print("=" * 78)
    s = sv["validation1_real_levels"]["summary"]
    c.check("solve rate on real levels", 100.00, 100 * s["solve_rate"], 0.01)
    c.check("false unsolvable verdicts", 0,
            len(sv["validation1_real_levels"]["false_unsolvable"]))
    c.check("solutions failing replay", 0,
            len(sv["validation1_real_levels"]["bad_replay"]))
    abl = sv["validation2_soundness_ablation"]
    c.check("ablation disagreements", 0,
            len(abl["disagree_status"]) + len(abl["disagree_length"]))
    c.check("player region mean", 19.1,
            sv["validation1_real_levels"]["player_region"]["mean"], 0.1)
    mb = sv["validation5_move_bound"]
    c.check("move-bound mean excess %", 40.5, 100 * mb["mean_relative_excess"], 0.5)
    c.check("move-bound exactly optimal %", 6.6,
            100 * mb["frac_exactly_optimal"], 0.5)
    c.check("move-bound violations", 0, len(mb["bound_violations"]))
    xc = sv["validation1_real_levels"]["cross_check"]
    c.check("our push <= their push frac", 1.0, xc["our_push_le_their_push_frac"], 0.001)
    c.check("our move >= their move frac", 1.0, xc["our_move_ge_their_move_frac"], 0.001)

    print()
    print("=" * 78)
    print("Section III: Dataset")
    print("=" * 78)
    c.check("has '*' (box on goal)", False, dv["check1_alphabet"]["has_box_on_goal_star"])
    c.check("has '+' (player on goal)", False,
            dv["check1_alphabet"]["has_player_on_goal_plus"])
    c.check("invariant violations", 0, dv["check2_invariants"]["n_violations"])
    c.check("test rows not replaying", 63,
            dv["check3_solution_units"]["test_split_wellformed_but_not_replaying"])
    c.check("test not-replaying %", 6.3,
            100 * dv["check3_solution_units"]["test_split_wellformed_but_not_replaying"]
            / 1000, 0.05)
    c.check("train dropped %", 0.73,
            100 * (1 - dv["check4_coverage"]["usable_fraction"]), 0.02)
    c.check("train/test collisions", 0, dv["check5_duplicates"]["train_test_collisions"])
    ml = dv["check4_coverage"]["move_length"]
    c.check("difficulty tercile lo", 26.0, ml["tercile_boundaries"][0], 0.01)
    c.check("difficulty tercile hi", 35.0, ml["tercile_boundaries"][1], 0.01)
    c.check("length p90", 47.0, ml["deciles"][9], 0.01)
    c.check("length max", 130, ml["max"])

    print()
    print("=" * 78)
    print("Section VII-A: Counting")
    print("=" * 78)
    c.check("transformer unconstrained valid %", 82.6, valid("transformer_unconstrained"))
    c.check("transformer unconstrained draws", 1024, drawn("transformer_unconstrained"))
    c.check("transformer constrained valid %", 100.0, valid("transformer_constrained"))
    c.check("vae argmax valid %", 0.0, valid("vae_argmax"), 0.001)
    c.check("vae argmax draws", 100000, drawn("vae_argmax"))
    c.check("vae sampled valid %", 1.7, valid("vae_sample"), 0.05)
    c.check("vae sampled draws", 29184, drawn("vae_sample"))
    c.check("gan raw valid %", 8.6, valid("gan_raw"))
    c.check("tf exactly-4 boxes %", 87.1,
            100 * tiles("transformer_unconstrained", "box", "exact_rate"))
    c.check("tf box mean", 3.87, tiles("transformer_unconstrained", "box", "mean"), 0.01)
    c.check("tf box sd", 0.34, tiles("transformer_unconstrained", "box", "std"), 0.01)
    c.check("vae argmax box mean", 1.38, tiles("vae_argmax", "box", "mean"), 0.01)
    c.check("vae argmax goal mean", 0.01, tiles("vae_argmax", "goal", "mean"), 0.01)
    c.check("vae sampled box mean", 3.95, tiles("vae_sample", "box", "mean"), 0.01)
    c.check("vae sampled goal mean", 4.05, tiles("vae_sample", "goal", "mean"), 0.01)
    c.check("vae sampled exactly-4 boxes %", 21.8,
            100 * tiles("vae_sample", "box", "exact_rate"))
    f = cond["transformer_constrained"]["forcing"]
    c.check("forcing fired %", 25.0, 100 * f["forced_sequence_rate"], 0.05)
    c.check("forced tokens/level", 0.378, f["mean_forced_tokens_per_level"], 0.001)

    print()
    print("=" * 78)
    print("Section VII-B/C: Solvability and repair")
    print("=" * 78)
    c.check("random solvable %", 0.0, solv("random_placement"), 0.001)
    c.check("rule-based solvable %", 0.4, solv("rule_based"))
    c.check("open room solvable %", 23.4, solv("open_room"))
    c.check("transformer constrained solvable %", 49.6, solv("transformer_constrained"))
    c.check("transformer unconstrained solvable %", 56.0, solv("transformer_unconstrained"))
    c.check("tf unconstrained overall %", 46.3,
            cond["transformer_unconstrained"]["solvable_overall"]["pct"])
    c.check("retrieval novel %", 0.0, cond["retrieval"]["novelty"]["novel_pct"], 0.001)
    c.check("retrieval NN distance", 0.0,
            cond["retrieval"]["novelty"]["nn_distance"]["mean"], 0.001)
    c.check("vae repaired solvable %", 76.6, solv("vae_repaired"))
    c.check("vae repaired diversity", 37.8, cond["vae_repaired"]["diversity"]["mean"], 0.05)
    c.check("real diversity reference", 38.8,
            ev["reference_distributions"]["diversity"]["mean"], 0.05)
    c.check("gan repaired solvable %", 40.2, solv("gan_repaired"))
    cmp1 = ev["comparisons"]["transformer_constrained__vs__open_room"]
    c.check("tf vs open room diff pp", 26.2, cmp1["diff_pct"])
    c.check("tf vs open room z", 8.60, cmp1["z"], 0.01)
    cmp2 = ev["comparisons"]["transformer_constrained__vs__transformer_unconstrained"]
    c.check("constrained vs unconstrained z", -2.03, cmp2["z"], 0.01)
    c.check("constrained vs unconstrained p", 0.043, cmp2["p_value"], 0.001)

    print()
    print("=" * 78)
    print("Section VII-D: Controllability")
    print("=" * 78)
    ct = cond["transformer_constrained"]["controllability"]
    c.check("tf density bin %", 88.8, ct["density"]["bin_accuracy"]["pct"])
    c.check("tf connectivity bin %", 80.0, ct["connectivity"]["bin_accuracy"]["pct"])
    c.check("tf clustering bin %", 76.3, ct["clustering"]["bin_accuracy"]["pct"])
    c.check("tf difficulty bin %", 43.3, ct["difficulty"]["bin_accuracy"]["pct"])
    c.check("tf length spearman", 0.484, ct["solution_length"]["spearman"], 0.001)
    c.check("tf length censoring %", 55.2,
            100 * ct["solution_length"]["censoring_rate"])
    cr = cond["real_boxoban"]["controllability"]
    c.check("real density bin %", 48.8, cr["density"]["bin_accuracy"]["pct"])
    c.check("real connectivity bin %", 53.2, cr["connectivity"]["bin_accuracy"]["pct"])
    c.check("real clustering bin %", 52.2, cr["clustering"]["bin_accuracy"]["pct"])
    c.check("real difficulty bin %", 31.0, cr["difficulty"]["bin_accuracy"]["pct"])

    print()
    print("=" * 78)
    print("Section VII-E/F: Ablation, seeds, temperature")
    print("=" * 78)
    c.check("distilgpt2 solvable %", 63.6, solv("distilgpt2_constrained"))
    c.check("distilgpt2 forcing %", 7.4,
            100 * cond["distilgpt2_constrained"]["forcing"]["forced_sequence_rate"], 0.05)
    cd = cond["distilgpt2_constrained"]["controllability"]
    c.check("distilgpt2 length spearman", 0.673, cd["solution_length"]["spearman"], 0.001)

    tr_t = load("train_transformer.json")
    tr_d = load("train_distilgpt2.json")
    c.check("transformer params", 10734336, tr_t["n_params"])
    c.check("distilgpt2 params", 43352064, tr_d["n_params"])
    c.check("transformer train s", 661, round(tr_t["train_time_s"]), 1)
    c.check("distilgpt2 train s", 4479, round(tr_d["train_time_s"]), 1)
    c.check("param ratio", 4.0, tr_d["n_params"] / tr_t["n_params"], 0.05)
    c.check("time ratio", 6.8, tr_d["train_time_s"] / tr_t["train_time_s"], 0.05)

    ss = seeds["summary"]
    c.check("seed solvable mean", 42.7, ss["solvable_given_valid_pct"]["mean"])
    c.check("seed solvable sd", 7.0, ss["solvable_given_valid_pct"]["sd"])
    c.check("seed diversity mean", 38.60, ss["diversity"]["mean"], 0.02)
    c.check("seed diversity sd", 0.31, ss["diversity"]["sd"], 0.02)
    c.check("seed val_loss sd", 0.005, ss["val_loss"]["sd"], 0.001)

    # -- three-seed pretraining ablation ----------------------------------
    dseeds = load("seed_variance_distilgpt2.json")
    ds = dseeds["summary"]
    tvals = ss["solvable_given_valid_pct"]["values"]
    dvals = ds["solvable_given_valid_pct"]["values"]
    c.check("distilgpt2 seed mean", 64.2, ds["solvable_given_valid_pct"]["mean"])
    c.check("distilgpt2 seed sd", 1.8, ds["solvable_given_valid_pct"]["sd"])
    c.check("ablation gap pp", 21.5,
            ds["solvable_given_valid_pct"]["mean"]
            - ss["solvable_given_valid_pct"]["mean"])
    c.check("distilgpt2 seed min", 62.8, min(dvals))
    c.check("distilgpt2 seed max", 66.2, max(dvals))
    c.check("transformer seed min", 34.6, min(tvals))
    c.check("transformer seed max", 47.0, max(tvals))
    c.check("ranges disjoint", True, min(dvals) > max(tvals))
    c.check("sd ratio", 4.0,
            ss["solvable_given_valid_pct"]["sd"]
            / ds["solvable_given_valid_pct"]["sd"], 0.05)
    c.check("distilgpt2 forcing mean %", 10.0, ds["forced_sequence_rate"]["mean"])
    c.check("distilgpt2 forcing sd", 3.4, ds["forced_sequence_rate"]["sd"])
    c.check("transformer forcing mean %", 33.7, ss["forced_sequence_rate"]["mean"])
    c.check("transformer forcing sd", 18.9, ss["forced_sequence_rate"]["sd"])

    import itertools

    import numpy as np
    from scipy import stats
    w = stats.ttest_ind(dvals, tvals, equal_var=False)
    c.check("Welch t", 5.12, float(w.statistic), 0.01)
    c.check("Welch p", 0.028, float(w.pvalue), 0.001)
    allv = list(dvals) + list(tvals)
    obs = np.mean(dvals) - np.mean(tvals)
    ge = sum(1 for comb in itertools.combinations(range(6), 3)
             if np.mean([allv[i] for i in comb])
             - np.mean([allv[i] for i in range(6) if i not in comb]) >= obs)
    c.check("exact permutation p", 0.05, ge / 20, 0.001)
    c.check("Levene p", 0.50, float(stats.levene(dvals, tvals).pvalue), 0.01)

    import glob
    tt = [json.load(open(f, encoding="utf-8"))["train_time_s"]
          for f in sorted(glob.glob(os.path.join(args.results,
                                                 "train_transformer*.json")))]
    dt = [json.load(open(f, encoding="utf-8"))["train_time_s"]
          for f in sorted(glob.glob(os.path.join(args.results,
                                                 "train_distilgpt2*.json")))]
    c.check("transformer mean train s", 573, round(float(np.mean(tt))), 1)
    c.check("distilgpt2 mean train s", 3897, round(float(np.mean(dt))), 1)
    c.check("mean time ratio", 6.8, float(np.mean(dt) / np.mean(tt)), 0.05)

    for temp, want in ((0.6, 61.2), (1.5, 34.8)):
        c.check(f"temperature {temp} solvable %", want,
                solv(f"transformer_constrained_t{temp}"))
    c.check("temperature 0.6 diversity", 35.8,
            cond["transformer_constrained_t0.6"]["diversity"]["mean"], 0.05)
    c.check("temperature 1.2 diversity", 39.0,
            cond["transformer_constrained_t1.2"]["diversity"]["mean"], 0.05)

    print()
    print("=" * 78)
    print("Section VIII: Failure analysis")
    print("=" * 78)
    ood = cond["transformer_constrained"]["ood"]["solution_length"]
    c.check("OOD requested mean", 115.8, ood["mean_requested"], 0.1)
    c.check("OOD achieved mean", 29.5, ood["mean_achieved"], 0.1)
    c.check("OOD spearman", -0.394, ood["spearman"], 0.001)
    c.check("OOD vae repaired achieved", 42.3,
            cond["vae_repaired"]["ood"]["solution_length"]["mean_achieved"], 0.1)
    c.check("OOD real achieved", 30.3,
            cond["real_boxoban"]["ood"]["solution_length"]["mean_achieved"], 0.1)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    bad = c.report()
    if bad:
        print(f"\n  {bad} claim(s) in the paper do not match the artifacts.")
        sys.exit(1)
    print("\n  Every quoted number matches its artifact.")


if __name__ == "__main__":
    main()
