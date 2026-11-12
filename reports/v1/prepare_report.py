"""Freeze existing experiment outputs and generate report tables/figures.

Run from any directory with Python, NumPy, matplotlib, and Pillow installed.
No experiments are trained or re-evaluated by this script.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE / "figures"
TABLE = HERE / "tables"
SNAP = HERE / "sources"
for directory in (FIG, TABLE, SNAP):
    directory.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def esc(value):
    return str(value).translate(str.maketrans({
        "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%",
        "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}",
    }))


def table(name, headers, rows, caption, label, spec=None):
    columns = spec or "l" + "r" * (len(headers) - 1)
    lines = [r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}",
             r"\begin{longtable}{" + columns + "}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule",
             r"\endfirsthead", r"\toprule", " & ".join(headers) + r" \\",
             r"\midrule\endhead", r"\bottomrule\endfoot"]
    lines += [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    lines += [r"\end{longtable}\endgroup", ""]
    (TABLE / (name + ".tex")).write_text("\n".join(lines), encoding="utf-8")


def model_key(name):
    return tuple(int(n) for n in name.split("_")[1:])


def score(value):
    return "---" if value is None else f"{value:.7f}"


manifest = {"snapshot_date": "2026-11-12", "source_files": [], "figures": []}
commit = subprocess.run(
    ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
    cwd=ROOT, check=True, capture_output=True, text=True,
).stdout.strip()
manifest["source_commit"] = commit
(TABLE / "snapshot.tex").write_text(r"\newcommand{\sourcecommit}{" + commit[:12] + "}\n", encoding="utf-8")

# Include every saved structured experiment output, plus prediction arrays.
paths = set(ROOT.glob("results_level*.json"))
paths.update((ROOT / "augment/aug_checkpoints").rglob("*.json"))
paths.update((ROOT / "mlp").rglob("*.json"))
paths.update((ROOT / "cnn_test").glob("*.json"))
paths.update((ROOT / "cnn_test").glob("*.csv"))
paths.update((ROOT / "results/ridge").glob("*.csv"))
paths.update((ROOT / "results/ridge").glob("*.npy"))
paths.update(ROOT.glob("*.ipynb"))
paths.update((ROOT / "augment").glob("*.ipynb"))
paths.update((ROOT / "cnn_test").glob("*.ipynb"))
paths.update((ROOT / "mlp").rglob("*.ipynb"))
paths.update(ROOT.glob("*.py"))
for directory in ("camels_pipeline", "augment", "mlp", "models"):
    paths.update((ROOT / directory).rglob("*.py"))
paths.update(ROOT / p for p in ["config.yaml", "README.md", "requirements.txt"])
for path in sorted(paths):
    rel = path.relative_to(ROOT)
    raw = path.read_bytes()
    manifest["source_files"].append({"path": rel.as_posix(), "bytes": len(raw),
                                     "sha256": hashlib.sha256(raw).hexdigest()})
    # Notebooks are captured as cell sources + textual outputs, without duplicating base64.
    if path.suffix == ".ipynb":
        nb = json.loads(raw)
        outputs = []
        for i, cell in enumerate(nb["cells"]):
            result = {"cell": i, "cell_type": cell["cell_type"],
                      "source": cell.get("source", []), "outputs": []}
            for output in cell.get("outputs", []):
                result["outputs"].append({k: v for k, v in output.items() if k != "data"} |
                    {"data": {k: v for k, v in output.get("data", {}).items() if not k.startswith("image/")}})
            outputs.append(result)
        target = SNAP / rel.with_suffix(".outputs.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        target = SNAP / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

image_sources = [
    ("ridge_cv", "results/ridge/cv_curves.png", "Ridge cross-validation curves"),
    ("ridge_predictions", "results/ridge/pred_vs_true.png", "Ridge map predictions at all levels"),
    ("ridge_size", "results/ridge/rmse_vs_train_size.png", "Ridge test error versus training size"),
    ("mlp_learning", "mlp/level1/mlp_learning_curves.png", "Initial MLP level-1 learning curves"),
    ("mlp_test1", "mlp/level1/level1_test_predictions.png", "MLP level-1 test predictions"),
    ("mlp_test2", "mlp/level2/level2_test_predictions.png", "MLP level-2 test predictions"),
    ("mlp_test3", "mlp/level3/level3_test_predictions.png", "MLP level-3 test predictions"),
    ("cnn_errors", "cnn_test/level3_test_errors.png", "Common level-3 map test errors"),
    ("cnn_predictions", "cnn_test/level3_test_predictions.png", "Common level-3 simulation predictions"),
    ("augmentation", "augment/Group_Comparison_Noise_Complete.png", "Noisy-test group averages"),
]
hashes = {}
for name, rel, caption in image_sources:
    raw = (ROOT / rel).read_bytes()
    shutil.copyfile(ROOT / rel, FIG / f"{name}.png")
    digest = hashlib.sha256(raw).hexdigest()
    hashes[digest] = name
    manifest["figures"].append({"asset": f"figures/{name}.png", "source": rel,
                                "sha256": digest, "caption": caption})

notebook_variants = []
for path in sorted(p for p in paths if p.suffix == ".ipynb"):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        for j, output in enumerate(cell.get("outputs", [])):
            encoded = output.get("data", {}).get("image/png")
            if not encoded:
                continue
            raw = base64.b64decode("".join(encoded))
            digest = hashlib.sha256(raw).hexdigest()
            if digest in hashes:
                continue
            name = "nb_" + path.stem + f"_c{i}_o{j}"
            (FIG / f"{name}.png").write_bytes(raw)
            hashes[digest] = name
            entry = {"asset": f"figures/{name}.png", "source": path.relative_to(ROOT).as_posix(),
                     "cell": i, "output": j, "sha256": digest}
            manifest["figures"].append(entry)
            notebook_variants.append(entry)
gallery = []
for entry in notebook_variants:
    path = entry["asset"]
    caption = (r"Embedded notebook rendering from \texttt{" + esc(entry["source"]) + "}, "
               f"cell {entry['cell']} (zero based). This preserves the notebook's saved version of the plot.")
    gallery += [r"\begin{figure}[p]\centering",
                r"\includegraphics[width=.92\linewidth,height=.72\textheight,keepaspectratio]{" + path + "}",
                r"\caption{" + caption + "}", r"\end{figure}\clearpage"]
(TABLE / "notebook_gallery.tex").write_text("\n".join(gallery), encoding="utf-8")

ridge = list(csv.DictReader((ROOT / "results/ridge/ridge_results.csv").open()))
rows = []
for r in ridge:
    rows.append([r["level"], r"$\Omega_m$" if r["parameter"] == "Omega_m" else r"$\sigma_8$",
                 f"{float(r['alpha']):.3f}", *[f"{float(r[k]):.4f}" for k in
                 ("MAE", "RMSE", "R2", "sim_MAE", "sim_RMSE", "sim_R2", "baseline_RMSE")]])
table("ridge", ["Level", "Target", r"$\alpha$", "MAE", "RMSE", r"$R^2$", "Sim MAE", "Sim RMSE", r"Sim $R^2$", "Base RMSE"],
      rows, "Complete Ridge held-out test results in original units.", "tab:ridge")

mlp = {level: read(f"mlp/level{level}/level{level}_test_results.json") for level in (1, 2, 3)}
rows = []
for level, d in mlp.items():
    for t, target in enumerate([r"$\Omega_m$", r"$\sigma_8$"]):
        m, s = d["map_metrics"], d["simulation_metrics"]
        rows.append([level, target, *[f"{x:.4f}" for x in (m['mae'][t], m['rmse'][t], s['mae'][t], s['rmse'][t], m['baseline_rmse'][t])],
                     f"{100*(1-m['rmse'][t]/m['baseline_rmse'][t]):.1f}\\%"])
table("mlp_test", ["Level", "Target", "Map MAE", "Map RMSE", "Sim MAE", "Sim RMSE", "Base RMSE", "Map gain"],
      rows, "Complete MLP held-out test metrics; a negative gain means worse than the training-mean baseline.", "tab:mlp-test")

rows, foldrows = [], []
comparisons = {}
for level in (1, 2, 3):
    d = read(f"mlp/level{level}/level{level}_config_comparison.json")
    comparisons[level] = d
    for name, s in d["summaries"].items():
        rows.append([level, esc(name), f"{s['score']:.6f}",
                     *[f"{x:.4f}" for x in s['pooled_rmse']],
                     *[f"{x:.4f}" for x in s['fold_std_rmse']],
                     ",".join(str(x) for x in s["best_epochs"])])
        for f in d["fold_results"][name]:
            m = f["simulation_metrics"]
            foldrows.append([level, esc(name), f["validation_fold"], f["best_epoch"],
                             f"{f['best_val_loss']:.4f}", *[f"{x:.4f}" for x in m['rmse']],
                             *[f"{x:.4f}" for x in m['mae']]])
table("mlp_cv", ["Level", "Configuration", "Score", r"RMSE $\Omega_m$", r"RMSE $\sigma_8$", r"SD $\Omega_m$", r"SD $\sigma_8$", "Best epochs"],
      rows, "All MLP configuration summaries. RMSE is pooled by simulation; SD is the fold RMSE standard deviation.", "tab:mlp-cv")
table("mlp_folds", ["Level", "Configuration", "Fold", "Epoch", "Std MSE", r"RMSE $\Omega_m$", r"RMSE $\sigma_8$", r"MAE $\Omega_m$", r"MAE $\sigma_8$"],
      foldrows, "All 45 MLP configuration/fold experiments, with simulation-level errors in original units.", "tab:mlp-folds")

comparison = list(csv.DictReader((ROOT / "cnn_test/level3_test_comparison.csv").open()))
rows = []
for r in comparison:
    name = "CNN-10" if "10 steps" in r["model"] else "CNN-16" if "16 steps" in r["model"] else r["model"]
    rows.append([name, r"$\Omega_m$" if r["target"] == "Omega_m" else r"$\sigma_8$",
                 *[f"{float(r[k]):.4f}" for k in ('RMSE', 'MAE', 'sim_RMSE', 'sim_MAE')],
                 f"{float(r['RMSE_reduction_%']):.1f}\\%"])
table("common_test", ["Model", "Target", "Map RMSE", "Map MAE", "Sim RMSE", "Sim MAE", "Map gain"],
      rows, "Common level-3 test comparison: 1,800 maps from 120 held-out simulations; each model was fitted on 7,200 development maps.", "tab:common")

all_scores = []
tagged = {}
for level in (1, 2, 3):
    tagged[level] = {}
    for tag in ("danial", "alen"):
        path = f"results_level{level}_{tag}.json"
        data = read(path)
        tagged[level].update(data)
        all_scores += [{"source": path, "condition": "clean-tagged", "level": level, "model": m, "validation_mse": s} for m, s in data.items()]
    noisy = {}
    for alpha, prefix in [("001", "aug01"), ("005", "aug05")]:
        noisy[alpha] = {}
        for group, tag in [("4-10", "danial"), ("12-16", "alen")]:
            path = f"augment/aug_checkpoints/{alpha}_{group}/{prefix}results_level{level}_yeradil_{tag}models.json"
            data = read(path)
            noisy[alpha].update(data)
            all_scores += [{"source": path, "condition": alpha, "level": level, "model": m, "validation_mse": s} for m, s in data.items()]
    names = sorted(set(tagged[level]) | set(noisy['001']) | set(noisy['005']), key=model_key)
    table(f"cnn_level{level}", ["Model ID", "Clean", r"$\alpha=0.01$", r"$\alpha=0.05$"],
          [[esc(m), score(tagged[level].get(m)), score(noisy['001'].get(m)), score(noisy['005'].get(m))] for m in names],
          f"All tagged CNN validation scores at level {level}; missing experiments are marked with a dash. MSE averages both unscaled targets.", f"tab:cnn{level}")
shared = {level: read(f"results_level{level}.json") for level in (1, 2, 3)}
table("cnn_shared", ["Model ID", "Level 1", "Level 2", "Level 3"],
      [[esc(m), *[score(shared[level].get(m)) for level in (1, 2, 3)]] for m in sorted(set().union(*(set(s) for s in shared.values())), key=model_key)],
      "Separate historical shared-file CNN scores. These are preserved rather than substituted for the tagged runs used by the final-selection workflow.", "tab:cnn-shared")
for level, data in shared.items():
    all_scores += [{"source": f"results_level{level}.json", "condition": "clean-shared", "level": level, "model": m, "validation_mse": s} for m, s in data.items()]
with (HERE / "cnn_scores.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_scores[0]));writer.writeheader();writer.writerows(all_scores)

# Parse numeric group results from the notebook's saved HTML table, not the chart pixels.
nb = read("augment/compare_results.ipynb")
groups = []
for output in nb['cells'][2]['outputs']:
    html = ''.join(output.get('data', {}).get('text/html', []))
    if '<table' not in html:
        continue
    for row in re.findall(r'<tr>(.*?)</tr>', html, re.S):
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.S)
        if cells and cells[0] in ('1% Noise', '5% Noise'):
            groups.append([len(groups)//2+1, cells[0], *map(float,cells[1:])])
assert len(groups) == 4, 'Expected both saved noise settings for both CNN groups'
table("noise_test", ["Group", "Noise", r"MAE $\Omega_m$", r"MAE $\sigma_8$", r"RMSE $\Omega_m$", r"RMSE $\sigma_8$"],
      [[r[0],esc(r[1]),*[f"{x:.6f}" for x in r[2:]]] for r in groups],
      "Saved level-3 noisy-test group averages (five model metrics averaged, not an ensemble). Notebook table precision is six decimals.", "tab:noise")

# Validate saved CNN prediction metrics without running a model or altering outputs.
cnn = read("cnn_test/level3_test_results.json")
target = np.asarray(cnn['map_targets'])
sim_target = np.asarray(cnn['simulation_targets'])
for name, d in cnn['models'].items():
    for predictions, true, metrics in [(d['map_predictions'],target,d['map_metrics']),
                                        (d['simulation_predictions'],sim_target,d['simulation_metrics'])]:
        error = np.asarray(predictions) - true
        np.testing.assert_allclose(np.sqrt((error**2).mean(axis=0)), metrics['rmse'], atol=2e-7)
        np.testing.assert_allclose(np.abs(error).mean(axis=0), metrics['mae'], atol=2e-7)
for level, d in mlp.items():
    pred, true = np.asarray(d['simulation_predictions']), np.asarray(d['simulation_targets'])
    np.testing.assert_allclose(np.sqrt(((pred-true)**2).mean(axis=0)), d['simulation_metrics']['rmse'], atol=2e-7)

# Additional report plots use only existing metrics/histories.
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for t, ax in enumerate(axes):
    targetname=['Omega_m','sigma_8'][t]
    ax.plot([810,3990,7200], [float(r['RMSE']) for r in ridge if r['parameter']==targetname], 'o-', label='Ridge / map')
    ax.plot([810,3990,7200], [mlp[l]['map_metrics']['rmse'][t] for l in (1,2,3)], 's-', label='MLP / map')
    ax.plot([810,3990,7200], [mlp[l]['simulation_metrics']['rmse'][t] for l in (1,2,3)], 's--', label='MLP / simulation')
    ax.set(xlabel='Development maps', ylabel='Held-out RMSE', title=[r'$\Omega_m$',r'$\sigma_8$'][t]);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(FIG/'spectral_comparison.pdf');plt.close(fig)

fig, axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for level, ax in zip((1,2,3), axes):
    chosen = mlp[level]['settings']['configuration_name']
    for f in comparisons[level]['fold_results'][chosen]:
        ax.plot(np.arange(1,len(f['val_history'])+1),f['val_history'],label=f"Fold {f['validation_fold']}")
    ax.set(xlabel='Epoch',ylabel='Standardized-target MSE',title=f'Level {level}: {chosen.replace("_"," ")}');ax.grid(alpha=.2)
axes[-1].legend(fontsize=8)
fig.savefig(FIG/'mlp_fold_histories.pdf');plt.close(fig)

fig, axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for level, ax in zip((1,2,3), axes):
    series=[('Clean',tagged[level])]
    for alpha,prefix in [('001','aug01'),('005','aug05')]:
        data={}
        for group,tag in [('4-10','danial'),('12-16','alen')]:
            data.update(read(f'augment/aug_checkpoints/{alpha}_{group}/{prefix}results_level{level}_yeradil_{tag}models.json'))
        series.append((f'Noise {"1%" if alpha=="001" else "5%"}',data))
    for title,data in series:
        depths=sorted({model_key(m)[0] for m in data})
        ax.plot(depths,[min(s for m,s in data.items() if model_key(m)[0]==depth) for depth in depths],'o-',label=title)
    ax.set(yscale='log',xlabel='Architecture step label',ylabel='Best saved validation MSE',title=f'Level {level}');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(FIG/'cnn_validation_summary.pdf');plt.close(fig)

# Inspection-only contact sheet of all original raster figures.
thumbs=[]
for entry in manifest['figures']:
    thumb=Image.new('RGB',(440,330),'white')
    image=Image.open(HERE/entry['asset']).convert('RGB');image.thumbnail((420,285))
    thumb.paste(image,((440-image.width)//2,25))
    ImageDraw.Draw(thumb).text((8,7),Path(entry['asset']).stem,fill='black')
    thumbs.append(thumb)
sheet=Image.new('RGB',(440*3,330*((len(thumbs)+2)//3)), '#e5e7eb')
for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%3)*440,(i//3)*330))
sheet.save(HERE/'figure_contact_sheet.jpg')

manifest['counts']={'cnn_validation_scores':len(all_scores),'original_raster_figures':len(manifest['figures']),
                    'structured_and_code_sources':len(manifest['source_files']),'mlp_configuration_folds':len(foldrows)}
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
table('figure_inventory',['Report asset','Original source'],
      [[r'\path{'+Path(e['asset']).name+'}',r'\path{'+e['source']+'}'+(f" (cell {e['cell']})" if 'cell' in e else '')] for e in manifest['figures']],
      'Inventory of all original saved raster graphs, including distinct notebook renderings.', 'tab:figures', 'p{.39\\linewidth}p{.55\\linewidth}')
print(json.dumps({'commit':commit,**manifest['counts'],'verified_saved_predictions':True},indent=2))
