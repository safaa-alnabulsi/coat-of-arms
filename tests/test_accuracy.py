from unittest import TestCase
from src.accuracy import Accuracy


pairs_pred_truth = [
    ['b a lion lion', 'b a lion passt guard', 80, 60],
    ['b a lion passt guard', 'b a lion passt guard', 100, 100],
    ['b a eagle', 'b a lion passt guard', 20, 0],
    ['A O lion rampant', 'A G lion rampant', 90, 100]
]

class AccuracyTest(TestCase):

    def test_get(self):
        for pair in pairs_pred_truth:
            predicted, truth, accuracy = pair[0], pair[1], pair[2]
            assert Accuracy(predicted, truth).get() == accuracy

    def test_get_charges_acc(self):
        for pair in pairs_pred_truth:
            predicted, truth, accuracy = pair[0], pair[1], pair[3]
            assert Accuracy(predicted, truth).get_charges_acc() == accuracy


