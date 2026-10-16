from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .config import ROOT, sha256_file, sha256_json
from .utils import task_complete, task_context


def _code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def _markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        _markdown(
            """
# Compression or Quantum Head

## Reproducible analysis for Article 09

This notebook reconstructs the audit summaries, performance tables, statistical comparisons, and every manuscript figure from immutable run artefacts. The experimental question is whether the dominant loss arises during the 384D-to-6D representation bottleneck or in the downstream classical/quantum decision head.
"""
        ),
        _code(
            """
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import SVG, display
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from article09.data import DatasetStore

FIG = ROOT / 'results' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='paper', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})

def save_figure(fig, stem):
    fig.savefig(FIG / f'{stem}.svg', bbox_inches='tight')
    fig.savefig(FIG / f'{stem}.png', bbox_inches='tight')
"""
        ),
        _markdown("## Environment and locked protocol"),
        _code(
            """
environment = json.loads((ROOT / 'environment' / 'hardware.json').read_text(encoding='utf-8'))
protocol = json.loads((ROOT / 'frozen' / 'protocol_v1.json').read_text(encoding='utf-8'))
pd.DataFrame({
    'Item': ['Python', 'PyTorch', 'CUDA', 'GPUs', 'Quantum backend', 'Final seeds'],
    'Value': [
        environment['python'].split()[0],
        environment['packages']['torch'],
        environment['torch_cuda'],
        ', '.join(environment['gpu_names']),
        protocol['selected_backend'],
        ', '.join(map(str, protocol['final_seeds']))
    ]
})
"""
        ),
        _markdown("## Figure 2 Experimental pipeline"),
        _code(
            """
drawio_source = FIG / 'figure_01_pipeline.drawio'
pipeline_svg = FIG / 'figure_01_pipeline.svg'
assert drawio_source.exists() and pipeline_svg.exists()
display(SVG(filename=str(pipeline_svg)))
print(f'Editable draw.io source: {drawio_source.relative_to(ROOT)}')
"""
        ),
        _markdown("## Dataset audit and sample montage"),
        _code(
            """
audit_rows = []
for dataset in ('cifar10','eurosat','pathmnist'):
    report = json.loads((ROOT/'data'/'audit'/f'{dataset}_audit_report.json').read_text(encoding='utf-8'))
    audit_rows.append({
        'Dataset': dataset,
        'Total decoded': report['total'],
        'Duplicate groups': report['duplicate_groups'],
        'Near-pair links': report['confirmed_near_pair_links'],
        'Quarantined': report['quarantined'],
        **{f'{k.title()} split': v for k,v in report['split_counts'].items()}
    })
audit_table = pd.DataFrame(audit_rows)
audit_table.to_csv(ROOT/'results'/'tables'/'dataset_audit_summary.csv', index=False)
audit_table
"""
        ),
        _code(
            """
store = DatasetStore(download=False)
datasets = ('cifar10','eurosat','pathmnist')
fig, axes = plt.subplots(3, 10, figsize=(15, 5.2))
for row_idx, dataset in enumerate(datasets):
    manifest = pd.read_csv(ROOT/'data'/'manifests'/f'{dataset}.csv')
    usable = manifest[(manifest.split=='train') & (manifest.label>=0)]
    samples = usable.groupby('label', sort=True).head(1).head(10)
    for col_idx, (_, row) in enumerate(samples.iterrows()):
        axes[row_idx,col_idx].imshow(store.image(dataset,row.source_split,int(row.source_index)))
        axes[row_idx,col_idx].set_title(str(row.class_name)[:15], fontsize=7)
        axes[row_idx,col_idx].axis('off')
    for col_idx in range(len(samples),10): axes[row_idx,col_idx].axis('off')
    axes[row_idx,0].set_ylabel(dataset, fontsize=9, weight='bold')
fig.suptitle('Representative samples selected by the analysis notebook', fontsize=13, weight='bold')
fig.tight_layout()
save_figure(fig, 'figure_02_dataset_samples')
plt.show()
"""
        ),
        _markdown("## Main performance results"),
        _code(
            """
test = pd.read_csv(ROOT/'results'/'metrics'/'all_test_metrics.csv')
validation = pd.read_csv(ROOT/'results'/'metrics'/'all_validation_metrics.csv')
summary = pd.read_csv(ROOT/'results'/'tables'/'main_performance_summary.csv')
display(summary.round(4))
"""
        ),
        _code(
            """
reduced = test[test.reducer!='full384']
means = reduced.groupby(['dataset','reducer','head']).macro_f1.mean().reset_index()
fig, axes = plt.subplots(1,3,figsize=(13.5,3.8),sharey=True)
for ax,(dataset,frame) in zip(axes,means.groupby('dataset')):
    pivot = frame.pivot(index='reducer',columns='head',values='macro_f1').reindex(['pca6','lda6','nca6','slp6'])
    sns.heatmap(pivot,annot=True,fmt='.3f',vmin=0,vmax=1,cmap='viridis',ax=ax,cbar=ax is axes[-1])
    ax.set_title(dataset); ax.set_xlabel('Head'); ax.set_ylabel('Reducer')
fig.suptitle('Mean test macro-F1 across ten seeds',fontsize=13,weight='bold')
fig.tight_layout()
save_figure(fig,'figure_03_macro_f1_heatmap')
plt.show()
"""
        ),
        _markdown("## Compression penalty and quantum-head gap"),
        _code(
            """
compression = pd.read_csv(ROOT/'results'/'statistics'/'compression_penalty.csv')
head_gaps = pd.read_csv(ROOT/'results'/'statistics'/'head_gaps.csv')
display(compression.round(4))
display(head_gaps.round(4))
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1,2,figsize=(12.5,4.2))
sns.barplot(data=compression,x='reducer',y='compression_penalty_mean',hue='dataset',ax=axes[0])
axes[0].axhline(0,color='black',linewidth=.8); axes[0].set_title('Compression penalty: RBF-384D minus RBF-6D')
axes[0].set_ylabel('Macro-F1 difference'); axes[0].set_xlabel('Reducer')
mlp_gap = head_gaps[head_gaps.comparison=='vqc_minus_mlp']
sns.barplot(data=mlp_gap,x='reducer',y='mean_difference',hue='dataset',ax=axes[1])
axes[1].axhline(0,color='black',linewidth=.8); axes[1].set_title('Quantum-head gap: VQC minus matched MLP')
axes[1].set_ylabel('Macro-F1 difference'); axes[1].set_xlabel('Reducer')
fig.tight_layout(); save_figure(fig,'figure_04_compression_and_head_gaps'); plt.show()
"""
        ),
        _markdown("## Representation geometry"),
        _code(
            """
geometry = pd.read_csv(ROOT/'results'/'tables'/'representation_diagnostics.csv')
vqc_mean = test[(test['head']=='vqc') & (test['reducer']!='full384')].groupby(['dataset','reducer']).macro_f1.mean().reset_index()
geometry_mean = geometry.groupby(['dataset','reducer'],as_index=False).mean(numeric_only=True)
geometry_plot = geometry_mean.merge(vqc_mean,on=['dataset','reducer'])
fig, axes = plt.subplots(1,2,figsize=(10.8,4.0))
sns.scatterplot(data=geometry_plot,x='trustworthiness_k10',y='macro_f1',hue='dataset',style='reducer',s=100,ax=axes[0])
sns.scatterplot(data=geometry_plot,x='fisher_ratio',y='macro_f1',hue='dataset',style='reducer',s=100,ax=axes[1],legend=False)
axes[0].set_title('Neighborhood preservation and VQC performance')
axes[1].set_title('Class separation and VQC performance')
fig.tight_layout(); save_figure(fig,'figure_05_representation_geometry'); plt.show()
"""
        ),
        _markdown("## Confusion matrices selected only by validation performance"),
        _code(
            """
ranked = (validation[validation.reducer!='full384']
          .groupby(['dataset','reducer','head'],as_index=False).macro_f1.mean()
          .sort_values(['dataset','macro_f1'],ascending=[True,False]).groupby('dataset').head(1))
selected_runs = []
figure_stems = {
    'cifar10': 'figure_06a_confusion_cifar10',
    'eurosat': 'figure_06b_confusion_eurosat',
    'pathmnist': 'figure_06c_confusion_pathmnist',
}
for _,choice in ranked.iterrows():
    candidates = validation[(validation['dataset']==choice['dataset'])&(validation['reducer']==choice['reducer'])&(validation['head']==choice['head'])].copy()
    representative_index = (candidates['macro_f1']-candidates['macro_f1'].mean()).abs().idxmin()
    seed_row = candidates.loc[representative_index]
    pred = pd.read_parquet(ROOT/'runs'/seed_row['run_id']/'predictions.parquet')
    matrix = confusion_matrix(pred.true_label,pred.predicted_label,normalize='true')
    # Display every normalized cell, including 0.00, so the published matrix is complete.
    annotations = np.vectorize(lambda value: f'{value:.2f}')(matrix)
    fig, ax = plt.subplots(figsize=(6.2,5.4))
    sns.heatmap(matrix,annot=annotations,fmt='',annot_kws={'fontsize':8.0},cmap='Blues',vmin=0,vmax=1,
                linewidths=.40,linecolor='white',square=True,ax=ax,cbar=True,
                cbar_kws={'shrink':0.82,'pad':0.025})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.subplots_adjust(top=0.98,bottom=0.12,left=0.12,right=0.93)
    save_figure(fig,figure_stems[choice['dataset']]); plt.show()
    selected_runs.append(seed_row['run_id'])
selected_runs
"""
        ),
        _markdown("## Calibration"),
        _code(
            """
fig, axes = plt.subplots(1,3,figsize=(12.4,3.7))
for ax,run_id in zip(axes,selected_runs):
    pred = pd.read_parquet(ROOT/'runs'/run_id/'predictions.parquet')
    prob_cols = [c for c in pred if c.startswith('prob_class_')]
    probs = pred[prob_cols].to_numpy(); confidence=probs.max(1); correct=(probs.argmax(1)==pred.true_label.to_numpy()).astype(int)
    observed, predicted = calibration_curve(correct,confidence,n_bins=10,strategy='uniform')
    ax.plot([0,1],[0,1],'--',color='gray'); ax.plot(predicted,observed,'o-')
    ax.set_title(run_id.split('-')[1]); ax.set_xlabel('Confidence'); ax.set_ylabel('Empirical accuracy'); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.suptitle('Reliability diagrams for validation-selected configurations',fontsize=12.5,weight='bold')
fig.tight_layout(); save_figure(fig,'figure_07_calibration'); plt.show()
"""
        ),
        _markdown("## Runtime and completion audit"),
        _code(
            """
runtime = test.groupby(['dataset','head'],as_index=False).fit_seconds.mean()
fig, ax = plt.subplots(figsize=(7.5,4.0))
sns.barplot(data=runtime,x='head',y='fit_seconds',hue='dataset',ax=ax)
ax.set_yscale('log'); ax.set_ylabel('Mean fit time (seconds, log scale)'); ax.set_xlabel('Head')
ax.set_title('Observed implementation runtime; not evidence of quantum speedup')
fig.tight_layout(); save_figure(fig,'figure_08_runtime'); plt.show()

json.loads((ROOT/'results'/'completion_audit.json').read_text(encoding='utf-8')) | {'runs': 'omitted from display'}
"""
        ),
        _markdown(
            """
## Interpretation boundary

The experiment supports claims about predictive behavior under a fixed six-dimensional information budget and analytic statevector simulation. It does not establish quantum advantage or quantum speedup. Cross-domain evidence is summarized through direction consistency and effect-size ranges; datasets are not treated as pooled independent replicates.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Article09 Quantum Environment",
        "language": "python",
        "name": "article09_quantum",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return notebook


def build_and_execute_notebook(config: dict, resume: bool = True) -> Path:
    report_manifest = ROOT / "results" / "report_manifest.json"
    input_hash = sha256_json(
        {
            "report": sha256_file(report_manifest),
            "code": sha256_file(Path(__file__)),
        }
    )
    output = ROOT / "notebooks" / "Article09_Reproducible_Research.ipynb"
    if resume and task_complete("notebook", input_hash) and output.exists():
        return output
    with task_context("notebook", input_hash):
        notebook = build_notebook()
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        client = NotebookClient(
            notebook,
            timeout=7200,
            kernel_name="article09_quantum",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        executed = client.execute()
        nbformat.write(executed, output)
    return output

