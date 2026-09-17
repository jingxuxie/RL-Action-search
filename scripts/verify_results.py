#!/usr/bin/env python3
"""Check recorded numerical claims and output integrity without oracle selection."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
s=json.loads((RES/'summary.json').read_text())
expected={
 'geometry_instances':400,'menus_enumerated':5840,'box_vertices_checked':6000,
 'exact_mdp_budget_configs':234,'offline_datasets':480,'offline_proposals':1440,
 'accepted_proposals':907,'near_optimal_certificates':445,
 'model_confidence_failures':1,'observed_false_accepts':0,
 'observed_false_nearoptimal_certificates':0,'linear_critic_configs':180,
 'rare_control_datasets':500,'rare_control_proposals':1500,
 'rare_harmful_proposals':625,'rare_accepted_proposals':301,'rare_false_accepts':0,
 'exposure_configs':180}
for key,value in expected.items():
    if s[key]!=value:
        raise AssertionError(f'{key}: expected {value}, recorded {s[key]}')
assert s['max_enumeration_error']<1e-10
assert s['max_envelope_to_robust_ratio']<=4+1e-10
for key,value in {'rank_backup_reduction_vs_generic':.052119527449617786,
                  'exposure_backup_reduction':.5273401297497683,
                  'linear_mean_refresh_gain':.007262746499011105}.items():
    assert math.isclose(s[key],value,rel_tol=1e-7,abs_tol=1e-9),(key,s[key],value)
manifest=json.loads((RES/'manifest.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((RES/name).read_bytes()).hexdigest()==digest,name
assert len(manifest)==12
assert len(pd.read_csv(RES/'forks.csv'))==2952
report={'status':'passed','scientific_csv_files':len(manifest),
        'checked_summary_values':len(expected)+5,
        'note':'Tests of recorded claims, not formal proof verification or independence claims.'}
(RES/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
