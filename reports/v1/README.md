# Project report v1

Open `report.pdf` for the compiled report. `report.tex` is the main LaTeX source.
The report consolidates the saved outputs at source commit
`64e6cecd88693badbc28390cb8a452866b0bec04`, without retraining models.

Build in WSL (Ubuntu with XeLaTeX and Python 3):

```bash
cd /mnt/d/myfiles/NU/csci494/upload/cosmoreg/reports/v1
bash build.sh
```

From PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/d/myfiles/NU/csci494/upload/cosmoreg/reports/v1/build.sh
```

The folder is self-contained for compilation. It uses Latin Modern fonts,
standard LaTeX packages, local images/tables, and an inline bibliography.
No shell escape, network access, BibTeX, or model dependencies are needed to build.
`build.sh` runs four XeLaTeX passes and checks the final `report.log` with
`verify_build.py`. The check fails on errors, warnings, missing characters,
unresolved references/citations, and overfull/underfull boxes.
`build_verification.json` records the result, PDF hash, and page count.
Individual compiler transcripts are kept in `build/`.

Evidence and generated files:

- `sources/`: frozen full-precision result files, prediction arrays, relevant
  implementation/configuration, and notebook cell sources/text outputs.
- `source_manifest.json`: original evidence paths and SHA-256 hashes.
- `cnn_scores.csv`: all 243 original CNN validation records with source paths.
- `figures/`: all 17 distinct original raster graphs and three derived vector plots.
- `tables/`: generated LaTeX tables, including exhaustive CNN and MLP search records.
- `figure_contact_sheet.jpg`: overview of the original raster assets for inspection.

`inspect_pdf.py` is an optional visual check using PyMuPDF and Pillow. It renders
page previews under `build/preview/` and checks for empty pages or content outside
the PDF page bounds. These packages are not required to compile the report.

To regenerate the evidence snapshot, tables, and derived plots from the full
repository, run `python reports/v1/prepare_report.py` from the repository root.
This requires NumPy, matplotlib, and Pillow in that Python environment. It uses
only existing outputs and checks saved prediction metrics without running models.
Rebuild the PDF afterward. The main narrative is maintained in `report.tex`.

Standalone graphs are included in the main discussion. Seven distinct notebook
renderings are preserved in the appendix; byte-identical duplicates are included
once. Validation scores, clean held-out tests, and noisy tests are distinguished,
as are map-level and simulation-level errors. The noise tables average individual
model metrics and do not describe an ensemble or clean-test augmentation gains.
