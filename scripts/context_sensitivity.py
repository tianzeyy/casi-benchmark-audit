"""Transparent named-label sensitivity for contextual nonmedical meanings."""
from pathlib import Path
import json,csv
W=Path(__file__).resolve().parents[1]
GROUPS={
 'AC::alternating current':'electrical terminology',
 'C4::(po box) c4':'postal designation',
 'CA::california':'geographic name',
 'CVS::customer, value, service':'commercial name',
 'DC::district of columbia':'geographic name',
 'FSH::fairview southdale hospital':'institution name',
 'IA::iowa':'geographic name',
 'IB::international baccalaureate':'education',
 'IT::information technology':'occupation or field',
 'LA::los angeles':'geographic name',
 'LA::louisiana':'geographic name',
 'MOM::mall of america:moa':'commercial name',
 'MP::military police':'occupation',
 'MS::master of science':'academic degree',
 'NA::narcotics anonymous':'organization name',
 'NA::not applicable':'general notation',
 'PD::police department':'organization name',
}
parts=json.loads((W/'inputs/reference/class_partition_summary.json').read_text())
core=json.loads((W/'results/core_test_predictions.json').read_text())
audit=json.loads((W/'results/coverage_audit_predictions.json').read_text())
key=lambda r:r['sf']+'::'+r['gold']
def summary(rs):
 return dict(n=len(rs),passing=sum(r['passes'] for r in rs),pass_rate=sum(r['passes'] for r in rs)/len(rs),accuracy=sum(r['correct'] for r in rs)/len(rs))
rows=[]
for k,category in GROUPS.items():
 if k not in parts:continue
 rs=[r for r in audit+core if key(r)==k]
 rows.append(dict(label=k,category=category,n_unique=parts[k]['n_unique'],partition=parts[k]['partition'],evaluated_n=len(rs),passing=sum(r['passes'] for r in rs)))
result=dict(definition='Explicit named labels in contextual nonmedical categories; the unchanged original candidate inventories and thresholds are used. This is a named-label sensitivity, not a relabelled clinical-only benchmark.',labels=rows,
 audit_contextual=summary([r for r in audit if key(r) in GROUPS]),audit_remaining=summary([r for r in audit if key(r) not in GROUPS]),
 core_contextual=summary([r for r in core if key(r) in GROUPS]),core_remaining=summary([r for r in core if key(r) not in GROUPS]))
(W/'results/contextual_meaning_sensitivity.json').write_text(json.dumps(result,indent=2))
with (W/'results/contextual_meaning_labels.csv').open('w',newline='') as f:
 d=csv.DictWriter(f,fieldnames=list(rows[0]));d.writeheader();d.writerows(rows)
print(json.dumps(result,indent=2))
