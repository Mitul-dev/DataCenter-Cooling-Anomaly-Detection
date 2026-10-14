"""Validation-only seed selection and uncertainty with servers as clusters."""
import numpy as np
from .protocol import constrained_policy_key


def choose_seed_policy(rows):
    feasible=[r for r in rows if constrained_policy_key(r) is not None]
    if feasible:
        return max(feasible,key=constrained_policy_key),'feasible'
    diagnostic=[r for r in rows if r['recall']>=.4]
    if not diagnostic:
        raise ValueError('No diagnostic policy reaches the frozen recall floor')
    return max(diagnostic,key=lambda r:(r['precision'],r['event_recall_observable'],
        -r['false_bursts_per_node_day'])),'diagnostic_only'


def release_gate(seed_records,primary_seed=42):
    """Release only if all declared seeds pass and primary fallback is feasible."""
    if sorted(r['seed'] for r in seed_records)!=[7,42,123]:
        raise ValueError('The release gate requires all three predefined seeds')
    if not all(r['status']=='feasible' and constrained_policy_key(r) is not None for r in seed_records):
        return None
    primary=next(r for r in seed_records if r['seed']==primary_seed)
    if constrained_policy_key(primary['global_fallback_validation']) is None:return None
    return primary


def node_bootstrap(rows,replicates=2000,seed=20261014):
    """Resample entire nodes; never treat overlapping endpoints as independent."""
    if len(rows)<2 or replicates<2:raise ValueError('At least two clusters and replicates are needed')
    fields=['tp','fp','fn','tn','false_bursts','scored_node_days','detected','episodes','observable_episodes']
    values=np.array([[r[k] for k in fields] for r in rows],dtype=np.float64)
    weights=np.random.default_rng(seed).multinomial(len(rows),np.full(len(rows),1/len(rows)),size=replicates)
    tp,fp,fn,tn,bursts,days,detected,episodes,observable=(weights@values).T
    def ratio(a,b):return np.divide(a,b,out=np.zeros_like(a),where=b>0)
    p=ratio(tp,tp+fp);r=ratio(tp,tp+fn)
    metrics={'precision':p,'recall':r,'false_positive_rate':ratio(fp,fp+tn),
        'false_bursts_per_node_day':ratio(bursts,days),'event_recall_all':ratio(detected,episodes),
        'event_recall_observable':ratio(detected,observable),'f0_5':ratio(1.25*p*r,.25*p+r)}
    return {k:{'lower_95':float(np.percentile(v,2.5)),'upper_95':float(np.percentile(v,97.5))} for k,v in metrics.items()}
