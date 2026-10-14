import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dccad.supervised import trailing_statistics,features_from_clean,preprocess_csv,QUALITY_NAMES
from dccad.data import BASE_FEATURES
from dccad.task3_inference import score_window


class SupervisedTests(unittest.TestCase):
    def test_tree_statistics_use_only_the_given_history(self):
        x=np.arange(4,dtype=np.float32)[None,:,None]*2+3
        stats=trailing_statistics(x)
        np.testing.assert_allclose(stats,[ [9.,6.,np.sqrt(5.),2.] ])

    def test_quality_flags_remember_missing_and_invalid_before_fill(self):
        raw=np.ones((2,22),np.float32);raw[1,0]=np.nan;raw[1,1]=-1
        filled=np.ones((2,22),np.float32);scaler=StandardScaler().fit(filled)
        out=features_from_clean(raw,filled,scaler,BASE_FEATURES+QUALITY_NAMES)
        np.testing.assert_allclose(out[1,-3:],[1/22,1/22,2/22])
        self.assertEqual(out.shape,(2,25))

    def test_inference_does_not_use_labels_or_future_rows(self):
        raw=np.ones((24,22),np.float32)
        pre={'scaler':StandardScaler().fit(raw),'features':BASE_FEATURES,'clip':10.,'eligibility_width':24}
        frame=pd.DataFrame(raw,columns=BASE_FEATURES);frame['label']=0
        expected=preprocess_csv(frame,pre)
        frame['label']=2
        np.testing.assert_array_equal(preprocess_csv(frame,pre),expected)
        frame.loc[0,'ambient']=np.nan
        with self.assertRaisesRegex(ValueError,'remain'):preprocess_csv(frame,pre)

    def test_inference_rejects_gap_and_out_of_order_context(self):
        raw=np.ones((24,22),np.float32)
        pre={'scaler':StandardScaler().fit(raw),'features':BASE_FEATURES,'clip':10.,'eligibility_width':24}
        frame=pd.DataFrame(raw,columns=BASE_FEATURES)
        frame['timestamp']=pd.date_range('2022-01-01',periods=24,freq='15min',tz='UTC')
        frame.loc[10,'timestamp']+=pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError,'cadence'):preprocess_csv(frame,pre)

    def test_saved_scaling_is_applied_without_refitting_on_upload(self):
        fitting=np.stack([np.zeros(22),np.ones(22)*2]).astype(np.float32)
        scaler=StandardScaler().fit(fitting)
        frame=pd.DataFrame(np.ones((24,22),np.float32)*100,columns=BASE_FEATURES)
        pre={'scaler':scaler,'features':BASE_FEATURES,'clip':10.,'eligibility_width':24}
        out=preprocess_csv(frame,pre)
        np.testing.assert_array_equal(out,np.ones((24,22))*10)
        np.testing.assert_array_equal(scaler.mean_,np.ones(22))

    def test_standalone_inference_rejects_mixed_servers_and_missing_timestamps(self):
        with self.assertRaisesRegex(ValueError,'timestamps'):
            score_window('unused',pd.DataFrame(np.ones((24,22)),columns=BASE_FEATURES))
        frame=pd.DataFrame({'timestamp':pd.date_range('2022-01-01',periods=24,freq='15min'),
            'node':[7]*23+[8]})
        with self.assertRaisesRegex(ValueError,'one server'):score_window('unused',frame)


if __name__=='__main__':unittest.main()
