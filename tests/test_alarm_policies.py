import unittest
import numpy as np
import pandas as pd
from dccad.alarm_policies import apply_rule,notifications,PolicyEvaluator
from dccad.evaluation import consecutive_alarm,metrics,incident_metrics


class AlarmPolicyTests(unittest.TestCase):
    def setUp(self):
        self.meta=pd.DataFrame({'node':[7,7,7,7,7,8,8,8],
            'time_ns':np.array([0,15,30,60,75,0,15,30])*60*10**9,
            'y_end':[False,True,True,False,False,False,True,False]})
        self.episodes=pd.DataFrame([{'node':7,'start_ns':15*60*10**9,'end_ns':45*60*10**9,'readings':2},
            {'node':8,'start_ns':15*60*10**9,'end_ns':30*60*10**9,'readings':1},
            {'node':8,'start_ns':60*60*10**9,'end_ns':75*60*10**9,'readings':1}])

    def test_consecutive_rule_matches_existing_evaluator(self):
        scores=np.array([1,1,1,1,1,1,1,1.])
        np.testing.assert_array_equal(apply_rule(self.meta,scores,.5,'consecutive_2'),
            consecutive_alarm(self.meta,scores,.5,2,cadence_minutes=15))

    def test_two_of_three_is_causal_and_resets_at_gaps_and_nodes(self):
        scores=np.array([1,0,1,1,1,1,1,1.])
        expected=[False,False,True,False,False,False,False,True]
        np.testing.assert_array_equal(apply_rule(self.meta,scores,.5,'two_of_three'),expected)
        changed=scores.copy();changed[-1]=0
        np.testing.assert_array_equal(apply_rule(self.meta,changed,.5,'two_of_three')[:-1],expected[:-1])

    def test_fast_metrics_match_reference_including_unobservable_incident(self):
        score=np.linspace(0,1,len(self.meta));engine=PolicyEvaluator(self.meta,self.episodes)
        engine.register_scores('sample',score)
        for pred in [np.zeros(8,bool),np.ones(8,bool),np.array([1,1,0,1,0,0,1,1],bool)]:
            actual=engine.evaluate(pred,'sample')
            reference={**metrics(self.meta.y_end,score,pred),
                **incident_metrics(self.meta,pred,self.episodes,cadence_minutes=15)[0]}
            for key,value in reference.items():
                if key=='cadence_minutes':continue
                if value is None:self.assertIsNone(actual[key])
                else:self.assertAlmostEqual(actual[key],value,msg=key)

    def test_notifications_do_not_modify_predictions_and_track_missed_incident(self):
        meta=pd.DataFrame({'node':[7]*5,'time_ns':np.arange(5)*15*60*10**9})
        pred=np.array([1,0,1,0,1],bool);original=pred.copy()
        out=notifications(meta,pred,60)
        np.testing.assert_array_equal(out,[True,False,False,False,True])
        np.testing.assert_array_equal(pred,original)

    def test_strict_threshold_supports_precision_recall_curve_ties(self):
        meta=self.meta.iloc[:3];score=np.array([.1,.2,.2],np.float32)
        threshold=float(np.nextafter(np.float64(np.float32(.2)),-np.inf))
        np.testing.assert_array_equal(apply_rule(meta,score,threshold),[False,True,True])


if __name__=='__main__':unittest.main()
