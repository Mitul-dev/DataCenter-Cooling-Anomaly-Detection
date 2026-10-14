import unittest
import numpy as np
import pandas as pd
from dccad.protocol import calendar_split, cohort_metadata, constrained_policy_key, TRAIN_END, VALIDATION_END
from dccad.data import CADENCE_NS, windows
from dccad.evaluation import metrics, consecutive_alarm, incident_metrics


class ProtocolTests(unittest.TestCase):
    def test_same_calendar_boundaries_for_different_servers(self):
        times=np.array([TRAIN_END-1,TRAIN_END,VALIDATION_END-1,VALIDATION_END])
        data={'time_ns':np.tile(times,2),'node':np.repeat([7,16],4)}
        np.testing.assert_array_equal(calendar_split(data),[0,1,1,2,0,1,1,2])

    def test_unknown_history_is_not_an_unknown_target(self):
        data={'node':np.full(6,3),'time_ns':np.arange(6)*CADENCE_NS,
              'label':np.array([-1,0,2,3,0,0])}
        meta=cohort_metadata(data,np.ones((6,2)),np.zeros(6),0,width=3)
        self.assertEqual(meta.row_index.tolist(),[2,4,5])
        self.assertEqual(meta.history_unknown_count.tolist(),[1,1,1])
        self.assertEqual(meta.y_end.tolist(),[True,False,False])

    def test_cohort_does_not_cross_gap_or_split(self):
        data={'node':np.full(6,3),'time_ns':np.array([0,1,3,4,5,6])*CADENCE_NS,
              'label':np.zeros(6)}
        meta=cohort_metadata(data,np.ones((6,1)),np.array([0,0,0,0,1,1]),0,width=3)
        self.assertEqual(len(meta),0)
        self.assertEqual(len(cohort_metadata(data,np.ones((6,1)),np.array([0,0,0,0,1,1]),1,width=3)),0)

    def test_fifteen_minute_persistence_and_exposure(self):
        meta=pd.DataFrame({'node':[7,7,7,7],'time_ns':np.array([0,15,45,60])*60*10**9})
        pred=consecutive_alarm(meta,np.ones(4),.5,persistence=2,cadence_minutes=15)
        np.testing.assert_array_equal(pred,[False,True,False,True])
        episodes=pd.DataFrame(columns=['node','start_ns','end_ns','readings'])
        report,_=incident_metrics(meta,pred,episodes,cadence_minutes=15)
        self.assertEqual(report['false_bursts'],2)
        self.assertAlmostEqual(report['scored_node_days'],1/24)
        self.assertAlmostEqual(report['false_bursts_per_node_day'],48)

    def test_false_positive_and_discovery_ratios_have_different_denominators(self):
        report=metrics([True,False,False,False],[.9,.8,.7,.1],[True,True,True,False])
        self.assertAlmostEqual(report['false_positive_rate'],2/3)
        self.assertAlmostEqual(report['false_discovery_ratio'],2/3)
        report=metrics([True,False,False,False,False],[.9,.8,.1,.1,.1],[True,True,False,False,False])
        self.assertEqual(report['false_positive_rate'],.25)
        self.assertEqual(report['false_discovery_ratio'],.5)
        self.assertIsNone(metrics([False],[.1],[False])['false_discovery_ratio'])

    def test_no_alarm_policy_cannot_pass_targets(self):
        row={'precision':1.,'recall':0.,'false_positive_rate':0.,
             'false_bursts_per_node_day':0.,'event_recall_observable':0.}
        self.assertIsNone(constrained_policy_key(row))
        row.update(precision=.4,recall=.5,event_recall_observable=.7)
        self.assertIsNotNone(constrained_policy_key(row))
        row['false_positive_rate']=.02
        self.assertIsNone(constrained_policy_key(row))

    def test_legacy_windows_support_previously_unmodeled_nodes(self):
        data={'node':np.full(4,3),'time_ns':np.arange(4)*CADENCE_NS,'label':np.zeros(4)}
        x,meta=windows(data,np.zeros((4,1),np.float32),np.zeros(4),0,width=3,stride=1)
        self.assertEqual(len(meta),2)
        self.assertEqual(x.shape,(2,3,1))


if __name__=='__main__':unittest.main()
