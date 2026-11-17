from pathlib import Path
import ast,json,csv,hashlib,warnings,re
warnings.filterwarnings('ignore',category=SyntaxWarning)
r=Path(__file__).resolve().parents[1]; errors=[]; counts={'python':0,'notebooks':0,'csv':0,'summaries':0}
for f in (r/'experiments').rglob('*'):
 try:
  if f.suffix=='.py': ast.parse(f.read_text(encoding='utf-8-sig')); counts['python']+=1
  elif f.suffix=='.ipynb':
   n=json.loads(f.read_text(encoding='utf-8-sig')); counts['notebooks']+=1
   for c in n['cells']:
    if c['cell_type']=='code':
     s=''.join(line+'\n' for line in ''.join(c['source']).splitlines() if not line.lstrip().startswith(('%','!'))); ast.parse(s)
     assert not c.get('outputs'), 'uncleared output'
  elif f.suffix=='.csv':
   with f.open(encoding='utf-8-sig',newline='') as h: assert next(csv.reader(h),None) is not None
   counts['csv']+=1
  if f.name=='summary.json':
   j=json.loads(f.read_text()); assert (f.parent/'history.csv').exists(); assert j['basis'] in ['random_relu','random_fourier']; counts['summaries']+=1
 except Exception as e: errors.append(f'{f.relative_to(r)}: {e}; text={getattr(e, 'text', '')!r}')
for f in (r/'tools').glob('*.py'): ast.parse(f.read_text(encoding='utf-8-sig'))
records=json.loads((r/'provenance.json').read_text(encoding='utf-8-sig')); assert all((r/x['path']).is_file() for x in records)
print(json.dumps({'counts':counts,'provenance_files':len(records),'errors':errors},indent=2)); assert not errors




