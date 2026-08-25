"""Tables generated *from* ``results/evaluation.json`` -- never typed by hand.

Every number in the paper traces back to a JSON artifact that carries its own
config, seed and git SHA (spec operating rule 5).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

DISPLAY_NAMES = {
    "random_placement": "Random placement",
    "open_room": "Open room",
    "rule_based": "Rule-based",
    "retrieval": "Retrieval",
    "gan_raw": "Conditional GAN (raw)",
    "gan_repaired": "Conditional GAN (repaired)",
    "vae_argmax": "Conditional VAE (argmax)",
    "vae_sample": "Conditional VAE (sampled)",
    "vae_repaired": "Conditional VAE (repaired)",
    "transformer_unconstrained": "Transformer (unconstrained)",
    "transformer_constrained": "Transformer (constrained)",
    "distilgpt2_constrained": "DistilGPT-2 (constrained, ablation)",
    "real_boxoban": "Real Boxoban levels",
}

LEARNED = {"gan_raw", "gan_repaired", "vae_argmax", "vae_sample", "vae_repaired",
           "transformer_unconstrained", "transformer_constrained",
           "distilgpt2_constrained"}

TRAIN_ARTIFACT = {
    "gan_raw": "train_gan.json", "gan_repaired": "train_gan.json",
    "vae_argmax": "train_vae.json", "vae_sample": "train_vae.json",
    "vae_repaired": "train_vae.json",
    "transformer_unconstrained": "train_transformer.json",
    "transformer_constrained": "train_transformer.json",
    "distilgpt2_constrained": "train_distilgpt2.json",
}


def _pct(d: Optional[Dict], key: str = "pct", nd: int = 1) -> str:
    if not d or d.get(key) is None:
        return "--"
    v = d[key]
    if v != v:  # NaN
        return "--"
    return f"{v:.{nd}f}"


def _ci(d: Optional[Dict], nd: int = 1) -> str:
    if not d or d.get("pct") is None or d["pct"] != d["pct"]:
        return "--"
    return (f"{d['pct']:.{nd}f} [{d['ci_lo_pct']:.{nd}f}, "
            f"{d['ci_hi_pct']:.{nd}f}]")


def _num(v, nd: int = 1, thousands: bool = False) -> str:
    if v is None:
        return "--"
    if thousands:
        return f"{int(v):,}"
    return f"{v:.{nd}f}"


def load_train_times(results_dir: str) -> Dict[str, Dict]:
    out = {}
    for cond, fname in TRAIN_ARTIFACT.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                out[cond] = json.load(fh)
    return out


def main_table_rows(ev: Dict, results_dir: str) -> List[Dict]:
    trains = load_train_times(results_dir)
    rows = []
    for name in ev["row_order"]:
        if name not in DISPLAY_NAMES:
            continue  # temperature-sweep rows live in their own table
        c = ev["conditions"][name]
        tr = trains.get(name, {})
        rows.append({
            "name": name,
            "display": DISPLAY_NAMES[name],
            "struct_valid": _ci(c["structural_validity"]),
            "solvable_all": _ci(c["solvable_overall"]),
            "solvable_valid": _ci(c["solvable_given_valid"]),
            "timeout": _pct(c["outcomes"]["timeout"]),
            "unsolvable": _pct(c["outcomes"]["unsolvable"]),
            "samples_drawn": _num(c["samples_drawn"], thousands=True),
            "novelty_nn": _num(
                c.get("novelty", {}).get("nn_distance", {}).get("mean"), 2),
            "novel_pct": _num(c.get("novelty", {}).get("novel_pct"), 1),
            "diversity": _num(c.get("diversity", {}).get("mean"), 2),
            "params": (_num(tr.get("n_params"), thousands=True)
                       if name in LEARNED else "--"),
            "train_time": (_num(tr.get("train_time_s"), 0)
                           if name in LEARNED and tr.get("train_time_s") else "--"),
            "gen_ms": _num(c.get("generation_time_per_sample_ms"), 2),
            "flag": c.get("small_denominator_flag") or c.get("notes"),
        })
    return rows


def markdown_main_table(ev: Dict, results_dir: str) -> str:
    rows = main_table_rows(ev, results_dir)
    head = ("| Model | Struct. valid % | Solvable % | Solvable \\| valid % | "
            "Timeout % | Samples drawn / 500 valid | Novel % | Novelty (NN dist) | "
            "Diversity | Params | Train time (s) |")
    sep = "|" + "---|" * 11
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['display']} | {r['struct_valid']} | {r['solvable_all']} | "
            f"{r['solvable_valid']} | {r['timeout']} | {r['samples_drawn']} | "
            f"{r['novel_pct']} | {r['novelty_nn']} | {r['diversity']} | "
            f"{r['params']} | {r['train_time']} |")

    ref = ev.get("reference_distributions", {})
    notes = [
        "",
        "Percentages carry Wilson 95% intervals.  **Structural validity and "
        "solvability are reported separately and never merged**: "
        "`Solvable %` is of all samples drawn, `Solvable | valid %` is of "
        "structurally valid samples only.",
        "",
        f"Reference distributions on {ref.get('n', '?')} held-out real levels: "
        f"nearest-neighbour distance "
        f"{_num(ref.get('novelty', {}).get('nn_distance', {}).get('mean'), 2)}, "
        f"diversity {_num(ref.get('diversity', {}).get('mean'), 2)}.  Novelty and "
        "diversity numbers are only interpretable against these.",
        "",
        "Distance is cell-wise Hamming over the 100 tiles; novelty is "
        "symmetry-aware over all 8 dihedral transforms.",
        "",
        "Single-seed unless stated otherwise; 3 seeds are run on the main "
        "transformer configuration only.",
    ]
    flags = [f"- **{r['display']}**: {r['flag']}" for r in rows if r["flag"]]
    if flags:
        notes += ["", "**Flags:**"] + flags
    return "\n".join(lines + notes)


def latex_main_table(ev: Dict, results_dir: str) -> str:
    rows = main_table_rows(ev, results_dir)
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Structural validity and solver-verified solvability across "
        r"model families. Percentages carry Wilson 95\% confidence intervals. "
        r"Structural validity and solvability are reported separately and never "
        r"merged. All rows are single-seed except the main transformer "
        r"configuration, which is run with three seeds.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Model & Valid \% & Solv. \% & Solv.\,$|$\,valid \% & Timeout \% & "
        r"Draws & Novel \% & Diversity & Params \\",
        r"\midrule",
    ]
    for r in rows:
        out.append(
            f"{r['display']} & {r['struct_valid']} & {r['solvable_all']} & "
            f"{r['solvable_valid']} & {r['timeout']} & {r['samples_drawn']} & "
            f"{r['novel_pct']} & {r['diversity']} & {r['params']} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out).replace("_", r"\_")


def markdown_controllability_table(ev: Dict) -> str:
    """Per attribute, never as one number (spec 10.2)."""
    lines = [
        "| Model | Difficulty bin | Density bin | Connectivity bin | "
        "Clustering bin | Length Spearman | Length censoring % | n |",
        "|" + "---|" * 8,
    ]
    for name in ev["row_order"]:
        c = ev["conditions"].get(name, {})
        ctrl = c.get("controllability")
        if not ctrl:
            continue
        sl = ctrl.get("solution_length", {})
        lines.append(
            f"| {DISPLAY_NAMES.get(name, name)} | "
            f"{_pct(ctrl.get('difficulty', {}).get('bin_accuracy'))} | "
            f"{_pct(ctrl.get('density', {}).get('bin_accuracy'))} | "
            f"{_pct(ctrl.get('connectivity', {}).get('bin_accuracy'))} | "
            f"{_pct(ctrl.get('clustering', {}).get('bin_accuracy'))} | "
            f"{_num(sl.get('spearman'), 3)} | "
            f"{_num(100 * sl['censoring_rate'] if sl.get('censoring_rate') is not None else None, 1)} | "
            f"{sl.get('n_with_achieved_length', '--')} |")
    lines += [
        "",
        "Wall density, connectivity and box clustering are computable directly "
        "from the grid, so a model that reproduces surface statistics will "
        "score well on them.  **Solution length is not computable without "
        "search**, so it is the only attribute that tests real understanding.",
        "",
        "Censoring % is the fraction of suite samples with no achieved length "
        "(invalid, proven unsolvable, or timed out).  The correlation is "
        "computed on the remaining subsample, which is biased toward easy "
        "levels -- the direction that inflates it.",
        "",
        "Achieved solution length uses the exact move-optimal solver "
        "(`cost_mode=\"moves\"`), matching the move units used in captions.",
    ]
    return "\n".join(lines)


def markdown_ood_table(ev: Dict) -> str:
    lines = ["| Model | OOD length Spearman | OOD censoring % | "
             "Mean requested | Mean achieved | n |", "|" + "---|" * 6]
    any_row = False
    for name in ev["row_order"]:
        c = ev["conditions"].get(name, {})
        ood = c.get("ood")
        if not ood:
            continue
        sl = ood.get("solution_length", {})
        any_row = True
        lines.append(
            f"| {DISPLAY_NAMES.get(name, name)} | {_num(sl.get('spearman'), 3)} | "
            f"{_num(100 * sl['censoring_rate'] if sl.get('censoring_rate') is not None else None, 1)} | "
            f"{_num(sl.get('mean_requested'), 1)} | "
            f"{_num(sl.get('mean_achieved'), 1)} | "
            f"{sl.get('n_with_achieved_length', '--')} |")
    if not any_row:
        return ""
    lines += [
        "",
        "Out-of-distribution requests ask for solution lengths of 90, 110 and "
        "150 moves.  Measured on the training split: p90 = 47, p99 = 67, "
        "p99.9 = 87, maximum = 130.  Failure to extrapolate is the expected "
        "result and populates the Failure Analysis section.",
    ]
    return "\n".join(lines)


def markdown_tile_count_table(ev: Dict) -> str:
    """Direct evidence for the counting-constraint claim."""
    lines = ["| Model | Boxes (mean +/- sd) | Exactly 4 boxes % | "
             "Goals (mean +/- sd) | Exactly 4 goals % | Players (mean) | "
             "Exactly 1 player % |", "|" + "---|" * 7]
    for name in ev["row_order"]:
        c = ev["conditions"].get(name, {})
        tc = c.get("tile_counts_raw_draws")
        if not tc or name not in DISPLAY_NAMES:
            continue
        b, g, p = tc["box"], tc["goal"], tc["player"]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {_num(b['mean'], 2)} +/- {_num(b['std'], 2)} | "
            f"{_num(100 * b['exact_rate'] if b['exact_rate'] is not None else None, 1)} | "
            f"{_num(g['mean'], 2)} +/- {_num(g['std'], 2)} | "
            f"{_num(100 * g['exact_rate'] if g['exact_rate'] is not None else None, 1)} | "
            f"{_num(p['mean'], 2)} | "
            f"{_num(100 * p['exact_rate'] if p['exact_rate'] is not None else None, 1)} |")
    lines += [
        "",
        "Computed over **raw draws**, valid or not: restricting to valid levels "
        "would define the counting failure away.  A valid level needs exactly "
        "one player, four boxes and four goals.",
    ]
    return "\n".join(lines)


def markdown_comparisons(ev: Dict) -> str:
    comps = ev.get("comparisons", {})
    if not comps:
        return ""
    lines = ["| Comparison (solvable \\| valid) | Difference (pp) | z | p |",
             "|---|---|---|---|"]
    for key, c in comps.items():
        if c.get("p_value") is None:
            continue
        a = DISPLAY_NAMES.get(c["a"], c["a"])
        b = DISPLAY_NAMES.get(c["b"], c["b"])
        lines.append(f"| {a} vs {b} | {c['diff_pct']:+.1f} | {c['z']:.2f} | "
                     f"{c['p_value']:.2e} |")
    return "\n".join(lines)


def markdown_temperature_table(ev: Dict) -> str:
    rows = []
    for name, c in ev["conditions"].items():
        if not name.startswith("transformer_constrained_t"):
            continue
        rows.append((c.get("temperature", name), c))
    if not rows:
        return ""
    rows.sort(key=lambda x: float(x[0]) if x[0] is not None else 0.0)
    lines = ["| Temperature | Solvable \\| valid % | Diversity | Novel % |",
             "|---|---|---|---|"]
    for temp, c in rows:
        lines.append(f"| {temp} | {_ci(c['solvable_given_valid'])} | "
                     f"{_num(c.get('diversity', {}).get('mean'), 2)} | "
                     f"{_num(c.get('novelty', {}).get('novel_pct'), 1)} |")
    return "\n".join(lines)


def write_all(evaluation_json: str, out_dir: str, results_dir: str) -> Dict[str, str]:
    with open(evaluation_json, "r", encoding="utf-8") as fh:
        ev = json.load(fh)
    os.makedirs(out_dir, exist_ok=True)

    parts = {
        "main_table.md": markdown_main_table(ev, results_dir),
        "main_table.tex": latex_main_table(ev, results_dir),
        "controllability_table.md": markdown_controllability_table(ev),
        "ood_table.md": markdown_ood_table(ev),
        "tile_counts_table.md": markdown_tile_count_table(ev),
        "comparisons_table.md": markdown_comparisons(ev),
        "temperature_table.md": markdown_temperature_table(ev),
    }
    written = {}
    for fname, text in parts.items():
        if not text:
            continue
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        written[fname] = path
    return written
