import unittest
import numpy as np
from model import generate_dataset, train_and_evaluate

class ModelTests(unittest.TestCase):
    def test_reproducible_dataset(self):
        x, y = generate_dataset()
        other, labels = generate_dataset()
        self.assertEqual(x.shape, (200, 64))
        np.testing.assert_array_equal(x, other)
        np.testing.assert_array_equal(y, labels)

    def test_trained_classifier_on_synthetic_users(self):
        accuracy, model = train_and_evaluate()
        self.assertGreaterEqual(accuracy, 0.9)
        self.assertLessEqual(accuracy, 1)
        self.assertEqual(len(model.classes_), 10)
