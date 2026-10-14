import unittest
import numpy as np
from dccad.data import CADENCE_NS
from dccad.preprocessing import causal_fill, sanitize, select_features, clean_inference


class PreprocessingTests(unittest.TestCase):
    def test_fill_resets_at_gap_split_and_four_missing_limit(self):
        x = np.array([[10.], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan],
                      [20.], [np.nan], [30.], [np.nan]], dtype=np.float32)
        times = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10]) * CADENCE_NS
        split = np.array([0]*9 + [1])
        out = causal_fill(x, np.full(10, 7), times, split)
        np.testing.assert_array_equal(out[1:5, 0], [10.]*4)
        self.assertTrue(np.isnan(out[[5, 7, 9], 0]).all())

    def test_cleaning_keeps_zeros_and_high_anomalies(self):
        x = np.array([[-71., 0., 5000.], [80., -1., np.inf]], dtype=np.float32)
        out = sanitize(x, ['p0_vdd_temp', 'fan0_0', 'total_power'])
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertEqual(out[0, 1], 0.)
        self.assertEqual(out[0, 2], 5000.)
        self.assertEqual(out[1, 0], 80.)
        self.assertTrue(np.isnan(out[1, 1:]).all())

    def test_signed_derived_changes_remain_valid(self):
        x=np.array([[-3.,-50.]],dtype=np.float32)
        np.testing.assert_array_equal(sanitize(x,['cpu_temp_change','power_change']),x)

    def test_feature_decisions_ignore_validation_and_test_values(self):
        train = np.array([[1., 2., 1.], [2., 4., 0.], [3., 6., 1.], [4., 8., 0.]])
        x = np.vstack([train, [[1., 7., 9.], [8., 3., 6.]]])
        data = {'features': np.array(['a', 'b', 'c']), 'label': np.zeros(6)}
        split = np.array([0, 0, 0, 0, 1, 2])
        selected, _, _ = select_features(data, x, x, split)
        changed = x.copy(); changed[4:] = [[np.nan]*3, [1e10]*3]
        selected2, _, _ = select_features(data, changed, changed, split)
        self.assertEqual(selected, ['a', 'c'])
        self.assertEqual(selected, selected2)

    def test_inference_cannot_fill_from_future(self):
        with self.assertRaisesRegex(ValueError, 'remain'):
            clean_inference(np.array([[np.nan], [30.]]), ['ambient'])
        out = clean_inference(np.array([[30.], [np.nan]]), ['ambient'])
        np.testing.assert_array_equal(out[:, 0], [30., 30.])


if __name__ == '__main__':
    unittest.main()
