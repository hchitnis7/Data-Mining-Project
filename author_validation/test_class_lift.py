"""Verify paired clusters and class-specific baselines for lift comparison."""
import unittest
import pandas as pd
from author_validation.compare_author_labels import class_lift_comparison


class LiftTests(unittest.TestCase):
    def test_new_labels_cannot_reselect_the_paired_cluster(self):
        df = pd.DataFrame({'cluster': [0]*4+[1]*4,
                           'future_subcat_old': ['CO']*3+['SR']+['CO']+['SR']*3,
                           'future_subcat_new': ['CO']+['SR']*3+['CO']*3+['SR']})
        row = class_lift_comparison(df, min_size=4).set_index('class').loc['CO']
        self.assertEqual(row.cluster, 0)
        self.assertEqual(row.new_best_cluster, 1)
        self.assertAlmostEqual(row.old_baseline, .5)
        self.assertAlmostEqual(row.new_baseline, .5)
        self.assertAlmostEqual(row.old_lift, .25)
        self.assertAlmostEqual(row.new_lift, -.25)
        self.assertAlmostEqual(row.lift_change_pp, -50)

    def test_baseline_changes_and_minimum_size(self):
        df = pd.DataFrame({'cluster': [0]*4+[1]*2,
                           'future_subcat_old': ['CO']*3+['SR']*3,
                           'future_subcat_new': ['CO']+['SR']*5})
        row = class_lift_comparison(df, min_size=4).set_index('class').loc['CO']
        self.assertEqual(row.cluster_size, 4)
        self.assertAlmostEqual(row.new_baseline, 1/6)
        self.assertAlmostEqual(row.new_lift, .25-1/6)
        missing = class_lift_comparison(df, min_size=10)
        self.assertTrue(missing.old_lift.isna().all())


if __name__ == '__main__':
    unittest.main()
