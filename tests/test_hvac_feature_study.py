import unittest
import numpy as np
import pandas as pd
from dccad.hvac_feature_study import fit_subset,transform_subset,window_features


class SensorSubsetTests(unittest.TestCase):
    def test_future_values_do_not_fit_scaling_or_imputation(self):
        raw=np.array([[1.,10.],[3.,20.],[999.,999.],[np.nan,np.nan]],dtype=np.float32)
        pre=fit_subset(raw,raw,np.array([True,True,False,False]),['a','b'])
        np.testing.assert_allclose(pre['median'],[2.,15.])
        np.testing.assert_allclose(pre['mean'],[2.,15.])
        transformed=transform_subset(raw,raw,pre)
        np.testing.assert_allclose(transformed[-1],[0.,0.])
        self.assertTrue(np.all(transformed[2]<=10.))

    def make_window(self):
        sensors=['outdoor_temp','cooling_valve','supply_temp','mixed_temp','outdoor_damper']
        raw=np.tile(np.arange(30,dtype=np.float32)[:,None],(1,5))
        pre=fit_subset(raw,raw,np.ones(30,dtype=bool),sensors)
        frame=pd.DataFrame(raw,columns=sensors)
        frame['timestamp']=pd.date_range('2026-01-01 08:00',periods=30,freq='min')
        frame['occupied']=1;frame['supply_fan_on']=1
        return frame,pre

    def test_five_sensor_input_ignores_labels_and_unrelated_channels(self):
        frame,pre=self.make_window();before=window_features(frame,pre)
        frame['Fault Detection Ground Truth']=1;frame['bias_c']=1000;frame['return_temp']=-999
        after=window_features(frame,pre)
        self.assertEqual(after.shape,(30,5));np.testing.assert_array_equal(before,after)

    def test_timestamp_and_equipment_boundaries_are_rejected(self):
        frame,pre=self.make_window();frame.loc[29,'timestamp']+=pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError,'contiguous'):window_features(frame,pre)
        frame,pre=self.make_window();frame['unit']=['A']*29+['B']
        with self.assertRaisesRegex(ValueError,'One unit'):window_features(frame,pre)
