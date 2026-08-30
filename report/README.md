# Course report: Generative AI Internal Assessment

`GenAI_IA_Report.pdf` (33 pages) is the deliverable. It covers the same work as
`paper/main.pdf` but in college-report form: cover page, contents, numbered
chapters, and far more explanation of the generative-modelling side.

## Rebuilding

```bash
python report/make_figures.py                     # regenerate the 5 report-only figures
cd report && pdflatex GenAI_IA_Report.tex         # run three times, for the ToC
```

`make_figures.py` reads every number it prints straight out of `results/`, and
for `r3_constrained_decoding` it loads `checkpoints/transformer.pt` and records
the model's real next-token probabilities at a decoding step where feasibility
forcing actually fires. Nothing in these figures is illustrative.

The report also embeds six figures from `figures/`, which are produced by the
main pipeline (`scripts/05_evaluate.py`).

## Note on MiKTeX

`pdflatex` fails on this machine with

```
MiKTeX cannot retrieve attributes for the directory
'C:\Program Files (x86)\Java\jdk-25\bin\java.exe\'
```

because two entries in `PATH` point at `.exe` files rather than directories, and
MiKTeX walks `PATH` when it needs to resolve a package. Stripping them for the
build works:

```bash
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.exe$' | paste -sd: -) \
  pdflatex GenAI_IA_Report.tex
```

The permanent fix is to remove those two entries from the user `PATH`.

## Report-only figures

| File | What it shows |
|---|---|
| `r1_sokoban_basics` | A real held-out level replayed under its 23-move solution, with a tile legend. |
| `r2_counting` | Autoregressive vs one-shot decoding, annotated with the measured exactly-4-boxes rates. |
| `r3_constrained_decoding` | The logit mask binding at step 96: 67.4% on floor before the mask, 100% on goal after. |
| `r4_seed_variance` | Solvability spread vs validation-loss spread across the same six training runs. |
| `r5_headline` | Every condition's solvability with Wilson intervals, ranked. |
