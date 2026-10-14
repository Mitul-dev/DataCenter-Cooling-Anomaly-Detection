import unittest
import numpy as np
import pandas as pd
from dccad.evaluation import consecutive_alarm, bursts, incident_metrics
from dccad.data import windows, CADENCE_NS


class EvaluationTests(unittest.TestCase):
    def test_duplicate_timestamp_cannot_hide_compensating_gap(self):
        data={'node':np.full(5,7), 'label':np.zeros(5,dtype=np.int8),
              'time_ns':np.array([0,1,1,3,4])*CADENCE_NS}
        x,meta=windows(data,np.zeros((5,1),dtype=np.float32),np.zeros(5,dtype=np.int8),0,width=4,stride=1)
        self.assertEqual(len(meta),0)

    def test_persistence_resets_at_gap_and_node_change(self):
        m = pd.DataFrame({"node": [7,7,7,7,8], "time_ns": np.array([0,30,90,120,150]) * 60 * 10**9})
        p = consecutive_alarm(m, np.ones(5), .5, persistence=2)
        np.testing.assert_array_equal(p, [False,True,False,True,False])
        self.assertEqual(bursts(m, np.ones(5,dtype=bool)), [(0,1),(2,3),(4,4)])

    def test_alarm_after_incident_does_not_detect_it(self):
        m = pd.DataFrame({"node": [7,7], "time_ns": np.array([0,30]) * 60 * 10**9})
        ep = pd.DataFrame([{"node":7,"start_ns":0,"end_ns":15*60*10**9,"readings":1}])
        r, detail = incident_metrics(m, np.array([False,True]), ep)
        self.assertEqual(r["detected"], 0)
        self.assertEqual(r["false_bursts"], 1)

    def test_unobservable_incidents_remain_in_total_denominator(self):
        m = pd.DataFrame({"node": [7], "time_ns": [0]})
        ep = pd.DataFrame([{"node":7,"start_ns":0,"end_ns":15*60*10**9,"readings":1},
                           {"node":7,"start_ns":15*60*10**9,"end_ns":30*60*10**9,"readings":1}])
        r, _ = incident_metrics(m, np.array([True]), ep)
        self.assertEqual(r["observable_episodes"], 1)
        self.assertEqual(r["event_recall_all"], .5)
        self.assertEqual(r["event_recall_observable"], 1.)


if __name__ == "__main__":
    unittest.main()
