"""Verify the Fourier run against markers in the final manuscript image."""
from pathlib import Path
import ast, csv, hashlib, json
R = Path(__file__).resolve().parents[1]
audit = json.loads((R/'docs/fourier_threshold_audit.json').read_text(encoding='utf-8'))
assert hashlib.sha256((R/audit['figure']).read_bytes()).hexdigest() == audit['figure_sha256']
x = audit['extraction']
observed = [(p-x['axes_left_px'])/(x['axes_right_px']-x['axes_left_px'])*x['epoch_max'] for p in x['event_pixel_centers']]
confirmed = audit['candidates']['7']['visible_event_epochs']
assert len(observed) == len(confirmed) == 17
error = max(abs(a-b) for a,b in zip(observed, confirmed))
assert error <= x['tolerance_epochs']
assert len(audit['candidates']['5']['visible_event_epochs']) != len(observed)
folder = R/'experiments/portfolio/results/adaptive_gamma_sgd_random_fourier_random_init'
paths = list(folder.glob('*/summary.json'))
assert len(paths) == 1
summary = json.loads(paths[0].read_text())
assert summary['adaptive_gamma']['saturation_threshold_ref'] == 7
f = paths[0].with_name('adaptive_gamma_events.csv')
assert hashlib.sha256(f.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == audit['candidates']['7']['event_csv_sha256_lf']
with f.open(newline='') as h:
    actual = [int(row['epoch_index']) for row in csv.DictReader(h) if int(row['epoch_index']) <= x['epoch_max']]
assert actual == confirmed
notebook = json.loads((R/'experiments/portfolio/04 Portfolio SGD Gamma Plots.ipynb').read_text(encoding='utf-8'))
assignments = [n for c in notebook['cells'] if c['cell_type']=='code' for n in ast.walk(ast.parse(''.join(c['source']))) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='adaptive_threshold_ref' for t in n.targets)]
assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == 7
runner = ast.parse((R/'tools/run_notebook.py').read_text(encoding='utf-8-sig'))
values = [k.value for n in ast.walk(runner) if isinstance(n,ast.Call) for k in n.keywords if k.arg=='adaptive_threshold_ref']
assert len(values) == 1 and ast.literal_eval(values[0]) == 7
with (R/'experiments/portfolio/savio/adaptive_gamma_threshold_grid.csv').open(encoding='utf-8-sig',newline='') as h:
    assert [float(row['threshold_ref']) for row in csv.DictReader(h)] == [7]
print('Confirmed threshold 7: all 17 manuscript markers match; max error %.3f gradient calls.' % error)
