import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dccad.data import BASE_FEATURES
from dccad.recovery import (SOURCE_COLUMNS,EXTRA_FEATURES,FEATURES,aggregate_extra,fit_extension,preprocess_recovery_csv)


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        base=np.ones((30,22),np.float32)
        extra=np.ones((30,len(EXTRA_FEATURES)),np.float32)*2
        data=dict(node=np.ones(30,int),time_ns=np.arange(30)*15*60*10**9,label=np.zeros(30,int))
        split=np.r_[np.zeros(24,int),np.ones(6,int)]
        return data,base,extra,split,StandardScaler().fit(base[:24])

    def test_incomplete_device_group_is_missing_not_a_false_low_maximum(self):
        cols={c:np.ones(2,np.float32) for c in SOURCE_COLUMNS}
        cols['gpu0_core_temp_max'][0]=np.nan
        out=aggregate_extra(cols)
        self.assertTrue(np.isnan(out[0,EXTRA_FEATURES.index('gpu_core_max')]))
        self.assertEqual(out[1,EXTRA_FEATURES.index('gpu_core_max')],1.)

    def test_training_fallback_ignores_later_values_and_labels(self):
        data,base,extra,split,scaler=self.fixture()
        _,pre,_=fit_extension(data,base,extra,split,scaler)
        extra[24:]=10000;data['label'][24:]=2
        _,changed,_=fit_extension(data,base,extra,split,scaler)
        np.testing.assert_array_equal(pre['extra_medians'],changed['extra_medians'])
        np.testing.assert_array_equal(pre['extra_scaler'].mean_,changed['extra_scaler'].mean_)

    def test_missing_flag_and_fallback_survive_without_dropping_an_endpoint(self):
        data,base,extra,split,scaler=self.fixture()
        extra[24:,0]=np.nan
        values,_,_=fit_extension(data,base,extra,split,scaler)
        self.assertEqual(values.shape,(30,len(FEATURES)))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[24,44],1.)
        self.assertAlmostEqual(values[24,-1],1/22)

    def test_uploaded_missing_extension_uses_saved_median_not_future_reading(self):
        data,base,extra,split,scaler=self.fixture();_,pre,_=fit_extension(data,base,extra,split,scaler)
        frame=pd.DataFrame(np.column_stack([base[:24],extra[:24]]),columns=BASE_FEATURES+EXTRA_FEATURES)
        frame['timestamp']=pd.date_range('2022-01-01',periods=24,freq='15min',tz='UTC')
        frame.loc[0,EXTRA_FEATURES[0]]=np.nan;frame.loc[1,EXTRA_FEATURES[0]]=100
        values=preprocess_recovery_csv(frame,pre)
        self.assertEqual(values[0,22],0.)
        self.assertEqual(values[0,44],1.)
        self.assertAlmostEqual(values[0,-1],1/22)

if __name__=='__main__':unittest.main()
