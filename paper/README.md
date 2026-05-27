# MiniMT3-Piano Paper

This directory contains the Chinese LaTeX report for the MiniMT3-Piano project.

Recommended build:

```bash
cd paper
latexmk -xelatex main.tex
```

Fallback build:

```bash
cd paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

The current paper uses finalized local metrics from:

- `outputs/eval_final_mainline/final_metrics.csv`
- `outputs/eval_final_mainline/final_metrics.json`
- `outputs/demo_compare/final_demo_report.json`

The latest `v13+v19` hybrid convergence files were not present locally when this draft was written:

- `outputs/eval_hybrid_converge/hybrid_metrics.json`
- `outputs/eval_hybrid_converge/hybrid_metrics.csv`
- `outputs/demo_compare/hybrid_demo_report.json`

If those files become available, update the AMT and demo tables before final submission.
