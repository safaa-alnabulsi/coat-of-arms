from unittest import TestCase
from src.accuracy import Accuracy


pairs_pred_truth = [
    # predicted, truth, shield accuracy, charge accuracy, total accuracy
    ['b a lion passt guard', 'b a lion passt guard', 1, 1, 1], # Both shield & chrage hit
    ['b a eagle', 'b a lion passt guard', 1, 0, 0.5],          # shield hits, chrage misses
    ['A O lion rampant', 'A G lion rampant', 1, 0.67, 0.835],  # shield hits, charge hit except its color
    ['A O lion rampant', 'A O lion', 1, 1, 1],                  # shield hits, charge hit with extra attribute in prediction
    ['A O lion', 'A O lion rampant', 1, 0.67, 0.835],         # shield hits, charge hit except its modifier
    ['b a a lion lion', 'b a lion passt guard', 1, 0.67, 0.835],     # multi object ===> 

]

class AccuracyTest(TestCase):

    def test_get_shield_acc(self):
        for pair in pairs_pred_truth:
            predicted, truth, accuracy = pair[0], pair[1], pair[2]
            assert Accuracy(predicted, truth).get_shield_acc() == accuracy

            
    def test_get_charges_acc(self):
        for pair in pairs_pred_truth:
            predicted, truth, accuracy = pair[0], pair[1], pair[3]
            assert Accuracy(predicted, truth).get_charges_acc() == accuracy

    def test_get(self):
        for pair in pairs_pred_truth:
            predicted, truth, accuracy = pair[0], pair[1], pair[4]
            assert Accuracy(predicted, truth).get() == accuracy
