"""Execute the curated notebooks with explicit working directories."""
from pathlib import Path
import argparse, json, os
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('experiment',choices=['portfolio-plots','multitask-preprocess','multitask-train','multitask-plots','multitask-smoke'])
p.add_argument('--basis',choices=['random_fourier','random_relu'],default='random_fourier')
a=p.parse_args()
if a.experiment=='multitask-train':
 import cyipopt  # Full evaluation must use the manuscript solver.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if a.experiment.startswith('portfolio'):
 folder=ROOT/'experiments/portfolio'; name='04 Portfolio SGD Gamma Plots.ipynb'
else:
 folder=ROOT/'experiments/multitask'
 name={'multitask-preprocess':'05 Multitask D_optimal Design Preprocessing.ipynb','multitask-train':'06 Multitask D_optimal Design.ipynb','multitask-smoke':'06 Multitask D_optimal Design.ipynb','multitask-plots':'07 Multitask D_optimal Design Plots.ipynb'}[a.experiment]
os.chdir(folder)
ns={'__name__':'__main__'}; out=ROOT/'figures'/a.experiment/a.basis; out.mkdir(parents=True,exist_ok=True)
fig_index=0
def save():
 global fig_index
 for num in plt.get_fignums():
  fig_index+=1; plt.figure(num).savefig(out/f'figure_{fig_index:02}.png',bbox_inches='tight',dpi=100)
 plt.close('all')
plt.show=save
for i,c in enumerate(json.loads((folder/name).read_text(encoding='utf-8'))['cells']):
 if c['cell_type']!='code': continue
 src=''.join(c['source'])
 if a.experiment=='multitask-smoke' and 'history.to_csv' in src:
  src='\n'.join(line for line in src.splitlines() if 'history.to_csv' not in line)
  exec(compile(src,str(folder/name)+f':cell{i}','exec'),ns)
  assert ns['history'].shape[0]>0
  print('Multitask reference smoke passed'); break
 exec(compile(src,str(folder/name)+f':cell{i}','exec'),ns)
 if a.experiment=='multitask-smoke': ns.update(T=2,n_eval=2)
 if a.experiment=='portfolio-plots' and 'results_dir_override = None' in src:
  # Threshold 7 matches the manuscript event markers (docs/FOURIER_THRESHOLD.md).
  ns.update(basis=a.basis,adaptive_threshold_ref=7,adaptive_threshold_alpha=0.1)
  if a.basis=='random_relu': ns.update(selected_gamma_indices=[0,4,6,9],adaptive_gamma_start=368.15625,adaptive_max_epochs_per_value=5000,max_epoch=10000)
  else: ns.update(selected_gamma_indices=[0,7,9,12],adaptive_gamma_start=89.25,adaptive_max_epochs_per_value=500,max_epoch=500)
save()
print('Finished',a.experiment,'figures:',out)

