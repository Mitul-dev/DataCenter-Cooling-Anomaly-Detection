import unittest
import numpy as np
import pandas as pd
from dccad.multisensor import fit_preprocessing, transform, decisions, classification_metrics


class MultiSensorTests(unittest.TestCase):
    def test_cleaning_uses_training_statistics_and_never_backfills(self):
        values=np.array([[1.,10.],[3.,20.],[999.,999.],[np.nan,np.nan]],dtype=np.float32)
        pre=fit_preprocessing(values,[True,True,False,False],['a','b'])
        np.testing.assert_allclose(pre['median'],[2.,15.])
        times=pd.date_range('2018-01-01 08:00',periods=4,freq='min').as_unit('ns').astype('int64').to_numpy()
        output=transform(values,['d1','d1','d2','d3'],times,pre)
        np.testing.assert_allclose(output[-1],[0.,0.])
        np.testing.assert_allclose(output[2],[10.,10.])
        self.assertTrue(np.isfinite(output).all())

    def test_sensor_persistence_resets_at_scenario_day_and_gap(self):
        times=np.array([0,1,2,3,4,6,7,8])*60*10**9
        meta=pd.DataFrame(dict(node=[1,1,1,2,2,2,2,2],time_ns=times))
        scores=np.tile([.02,.92,.06],(8,1))
        pred=decisions(meta,scores,[.5,.5],'two_of_three')
        np.testing.assert_array_equal(pred,[0,0,1,0,0,0,0,1])

    def test_two_eligible_sensors_choose_larger_score(self):
        meta=pd.DataFrame(dict(node=[1,1,1],time_ns=np.arange(3)*60*10**9))
        p=np.tile([.1,.4,.5],(3,1))
        np.testing.assert_array_equal(decisions(meta,p,[.3,.3]),[0,0,2])
        with self.assertRaisesRegex(ValueError,'normalized'):
            decisions(meta,p*2,[.3,.3])

    def test_fault_detection_does_not_equal_correct_sensor_identification(self):
        y=np.array([0,0,1,1,2,2]);pred=np.array([0,1,2,1,1,2])
        p=np.full((6,3),.1);p[np.arange(6),pred]=.8
        result=classification_metrics(y,pred,p,['healthy','temperature','pressure'])
        self.assertEqual(result['healthy_fpr'],.5)
        self.assertEqual(result['fault_detection_recall'],1.)
        self.assertEqual(result['per_class'][1]['recall'],.5)
        self.assertEqual(result['per_class'][2]['recall'],.5)
        self.assertEqual(result['confusion_matrix'],[[1,1,0],[0,1,1],[0,1,1]])
