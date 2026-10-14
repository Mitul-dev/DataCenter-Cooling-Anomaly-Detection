import unittest
from dccad.robustness import choose_seed_policy,release_gate,node_bootstrap


class RobustnessTests(unittest.TestCase):
    def row(self,seed=42,**changes):
        r=dict(seed=seed,precision=.4,recall=.5,false_positive_rate=.005,
            false_bursts_per_node_day=.05,event_recall_observable=.7,f0_5=.42,status='feasible')
        r.update(changes);return r

    def test_release_requires_every_seed_and_global_fallback(self):
        rows=[self.row(s) for s in [7,42,123]]
        rows[1]['global_fallback_validation']=self.row()
        self.assertEqual(release_gate(rows)['seed'],42)
        rows[0]['recall']=.1
        self.assertIsNone(release_gate(rows))
        rows[0]['recall']=.5;rows[1]['global_fallback_validation']['false_positive_rate']=.1
        self.assertIsNone(release_gate(rows))

    def test_weak_policy_remains_diagnostic_and_no_alarm_cannot_win(self):
        rows=[self.row(precision=.01,recall=.5),self.row(precision=.02,recall=.4),self.row(precision=0,recall=0)]
        chosen,status=choose_seed_policy(rows)
        self.assertEqual(chosen['precision'],.02);self.assertEqual(status,'diagnostic_only')
        with self.assertRaises(ValueError):choose_seed_policy([self.row(precision=0,recall=0)])

    def test_node_bootstrap_preserves_identical_cluster_rates(self):
        rows=[dict(tp=2,fp=3,fn=2,tn=93,false_bursts=1,scored_node_days=10,detected=1,episodes=2,observable_episodes=2)]*3
        ci=node_bootstrap(rows,replicates=100,seed=1)
        self.assertAlmostEqual(ci['precision']['lower_95'],.4)
        self.assertAlmostEqual(ci['precision']['upper_95'],.4)
        self.assertAlmostEqual(ci['false_bursts_per_node_day']['lower_95'],.1)

    def test_missing_seed_cannot_be_silently_omitted(self):
        with self.assertRaises(ValueError):release_gate([self.row(7),self.row(42)])

if __name__=='__main__':unittest.main()
