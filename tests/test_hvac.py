import unittest
import numpy as np
import pandas as pd
from dccad.hvac import RAW_FEATURES,augment,causal_clean,fit_preprocessor,transform,uploaded_features


class HVACPreparationTests(unittest.TestCase):
    def test_negative_outdoor_temperature_is_valid(self):
        raw=np.ones((2,13),np.float32);raw[:,2]=-20
        clean,filled=causal_clean(raw,['a','a'],np.arange(2)*60*10**9)
        np.testing.assert_array_equal(clean[:,2],[-20,-20])

    def test_fill_resets_at_days_and_has_no_future_access(self):
        raw=np.array([[2.],[np.nan],[np.nan],[4.]],np.float32)
        _,filled=causal_clean(raw,['a','a','b','b'],np.arange(4)*60*10**9)
        self.assertEqual(filled[1,0],2);self.assertTrue(np.isnan(filled[2,0]))

    def test_sensor_consistency_features_do_not_use_labels(self):
        raw=np.ones((1,13),np.float32);raw[0,0]=60;raw[0,1]=55
        self.assertEqual(augment(raw)[0,13],5.)
        self.assertEqual(augment(raw).shape,(1,19))

    def test_upload_requires_operating_context_and_retains_missing_flags(self):
        raw=np.ones((30,13),np.float32)
        pre=fit_preprocessor(raw,raw,raw,np.ones(30,bool))
        frame=pd.DataFrame(raw,columns=RAW_FEATURES)
        frame['timestamp']=pd.date_range('2017-01-01 06:00',periods=30,freq='1min')
        frame['occupied']=1;frame['supply_fan_on']=1
        self.assertTrue(np.isfinite(uploaded_features(frame,pre)).all())
        frame.loc[0,'outdoor_temp']=np.nan
        self.assertEqual(uploaded_features(frame,pre)[0,21],1.)
        frame.loc[0,'occupied']=0
        with self.assertRaisesRegex(ValueError,'occupied'):uploaded_features(frame,pre)

if __name__=='__main__':unittest.main()
