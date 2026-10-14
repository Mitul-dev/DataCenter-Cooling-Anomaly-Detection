"""Causal alarm policies and efficient, gap-aware validation evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def segment_starts(meta, cadence_minutes=15):
    if cadence_minutes <= 0:
        raise ValueError('Cadence must be positive')
    nodes=meta.node.to_numpy();times=meta.time_ns.to_numpy()
    if not len(times):return np.zeros(0,dtype=bool)
    if np.any((nodes[1:]==nodes[:-1])&(times[1:]<=times[:-1])):
        raise ValueError('Metadata must be chronological with unique timestamps within a node')
    return np.r_[True,(nodes[1:]!=nodes[:-1])|(np.diff(times)!=cadence_minutes*60*10**9)]


def apply_rule(meta, score, thresholds, rule='immediate', cadence_minutes=15):
    scores=np.asarray(score,dtype=np.float64)
    if scores.shape!=(len(meta),) or not np.isfinite(scores).all():
        raise ValueError('Finite one-dimensional scores must align with metadata')
    if isinstance(thresholds,dict):
        limit=np.array([thresholds[int(n)] for n in meta.node],dtype=np.float64)
    else:limit=float(thresholds)
    high=scores>limit
    reset=segment_starts(meta,cadence_minutes)
    if rule=='immediate':return high
    if not len(high):return high
    previous=np.r_[False,high[:-1]]
    if rule=='consecutive_2':return high&previous&~reset
    if rule=='two_of_three':
        before_previous=np.r_[False,False,high[:-2]][:len(high)]
        previous_reset=np.r_[True,reset[:-1]]
        complete=~reset&~previous_reset
        count=high.astype(np.int8)+previous.astype(np.int8)+before_previous.astype(np.int8)
        return complete&(count>=2)
    raise ValueError('Unknown alarm rule')


def notifications(meta,pred,cooldown_minutes=60,cadence_minutes=15):
    """Notify on an alarm's rising edge; never alter raw reading decisions."""
    pred=np.asarray(pred,dtype=bool)
    if pred.shape!=(len(meta),) or cooldown_minutes<0:
        raise ValueError('Aligned predictions and nonnegative cooldown are required')
    reset=segment_starts(meta,cadence_minutes)
    rising=pred&(reset|~np.r_[False,pred[:-1]]) if len(pred) else pred
    out=np.zeros(len(pred),dtype=bool);last=None
    times=meta.time_ns.to_numpy()
    for i in range(len(pred)):
        if reset[i]:last=None
        if rising[i] and (last is None or times[i]-last>=cooldown_minutes*60*10**9):
            out[i]=True;last=times[i]
    return out


class PolicyEvaluator:
    """Same incident definitions as evaluation.py; preprocess episode membership once."""
    def __init__(self,meta,episodes,cadence_minutes=15):
        self.meta=meta;self.episodes=episodes.reset_index(drop=True)
        self.cadence=cadence_minutes;self.reset=segment_starts(meta,cadence_minutes)
        self.node=meta.node.to_numpy();self.time=meta.time_ns.to_numpy()
        self.y=meta.y_end.to_numpy(dtype=bool)
        self.ep_id=np.full(len(meta),-1,dtype=np.int32)
        self.opportunities=np.zeros(len(episodes),dtype=np.int64)
        for i,ep in self.episodes.iterrows():
            inside=(self.node==ep.node)&(self.time>=ep.start_ns)&(self.time<ep.end_ns)
            if (self.ep_id[inside]>=0).any():raise ValueError('Incident intervals must not overlap')
            self.ep_id[inside]=i;self.opportunities[i]=inside.sum()
        if not np.array_equal(self.y,self.ep_id>=0):
            raise ValueError('Endpoint targets must agree with original incident intervals')
        self.node_days=len(meta)*cadence_minutes/1440
        self.score_metrics={}

    def register_scores(self,key,score):
        scores=np.asarray(score,dtype=np.float64)
        if scores.shape!=self.y.shape or not np.isfinite(scores).all():
            raise ValueError('Scores must be aligned and finite')
        self.score_metrics[key]={'average_precision':float(average_precision_score(self.y,scores)) if self.y.any() else None,
                                'roc_auc':float(roc_auc_score(self.y,scores)) if len(np.unique(self.y))==2 else None}

    def evaluate(self,pred,score_key):
        pred=np.asarray(pred,dtype=bool)
        if pred.shape!=self.y.shape:raise ValueError('Predictions must align with the cohort')
        tp=int((pred&self.y).sum());fp=int((pred&~self.y).sum())
        fn=int((~pred&self.y).sum());tn=int((~pred&~self.y).sum())
        p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
        starts=np.flatnonzero(pred&(self.reset|~np.r_[False,pred[:-1]]))
        next_reset=np.r_[self.reset[1:],True]
        ends=np.flatnonzero(pred&(next_reset|~np.r_[pred[1:],False]))
        hits=np.r_[0,np.cumsum(pred&self.y)]
        true_bursts=int(((hits[ends+1]-hits[starts])>0).sum())
        false_bursts=len(starts)-true_bursts
        first=np.full(len(self.episodes),np.iinfo(np.int64).max,dtype=np.int64)
        hit=np.flatnonzero(pred&self.y)
        if len(hit):np.minimum.at(first,self.ep_id[hit],self.time[hit])
        found=first!=np.iinfo(np.int64).max
        delay=(first[found]-self.episodes.start_ns.to_numpy(dtype=np.int64)[found])/(60*10**9)
        detected=int(found.sum());observable=int((self.opportunities>0).sum())
        return {**self.score_metrics[score_key],'precision':p,'recall':r,
            'f1':2*p*r/(p+r) if p+r else 0.,'f0_5':1.25*p*r/(.25*p+r) if .25*p+r else 0.,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None,
            'false_discovery_ratio':fp/(tp+fp) if tp+fp else None,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'episodes':len(self.episodes),'observable_episodes':observable,
            'detected':detected,'event_recall_all':detected/len(self.episodes) if len(self.episodes) else 0.,
            'event_recall_observable':detected/observable if observable else 0.,
            'alarm_bursts':len(starts),'false_bursts':false_bursts,
            'burst_precision':true_bursts/len(starts) if len(starts) else 0.,
            'scored_node_days':self.node_days,
            'false_bursts_per_node_day':false_bursts/self.node_days if self.node_days else None,
            'median_delay_minutes':float(np.median(delay)) if len(delay) else None}
