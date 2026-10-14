"""Check the evaluation boundary: test labels must not choose clusters."""
import unittest
import numpy as np
from compare import evaluate


class EvaluationTests(unittest.TestCase):
    def test_mapping_and_best_cluster_are_selected_on_training_only(self):
        train = np.array(['astro-ph.CO']*3 + ['astro-ph.SR']*3)
        clusters = np.array([0, 0, 0, 1, 1, 1])
        # Opposite test labels would yield perfect accuracy if test data chose mapping.
        test = np.array(['astro-ph.SR', 'astro-ph.SR', 'astro-ph.CO', 'astro-ph.CO'])
        test_clusters = np.array([0, 0, 1, 1])
        scores, rows = evaluate(train, test, clusters, test_clusters, 2)
        self.assertEqual(scores['test_accuracy'], 0)
        co = next(r for r in rows if r['class'] == 'astro-ph.CO')
        self.assertEqual(co['cluster'], 0)
        self.assertEqual(co['test_concentration'], 0)
        self.assertEqual(co['test_class_coverage'], 0)

    def test_no_large_enough_cluster_is_reported_as_missing(self):
        labels = np.array(['astro-ph.CO', 'astro-ph.SR'])
        scores, rows = evaluate(labels, labels, np.array([0, 1]), np.array([0, 1]), 50)
        self.assertEqual(scores['test_accuracy'], 1)
        self.assertTrue(all(r['cluster'] is None for r in rows))


if __name__ == '__main__':
    unittest.main()
