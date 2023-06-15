from unittest import TestCase
from src.accuracy import Accuracy, WEIGHT_MAP


pairs_pred_truth = [
    # predicted, truth, shield accuracy, charge accuracy, total accuracy
    ['b a lion passt guard', 'b a lion passt guard', 1, (1, 1), 1], # Both shield & chrage hit
    ['b a lion passt guard', 'b a lion passt guard & border', 1, (1, 1), 1], # Both half shield & chrage hit
    ['b a eagle', 'b a lion passt guard', 1, (0, 1), 0.67],          # shield hits, chrage misses
    ['A O lion rampant', 'A G lion rampant', 1, (1, 0), 0.67],  # shield hits, charge hit except its color
    ['A O lion rampant', 'A O lion', 1, (1, 1), 1],                  # shield hits, charge hit with extra attribute in prediction
    ['A O lion', 'A O lion rampant', 1, (0.5, 1), 0.83],         # shield hits, charge hit except its modifier
    ['b a a lion lion', 'b a lion passt guard', 1, (0.5, 1), 0.83],  # multi object ===> two predicted for one 
    
    # Keep those two: this example works when using get_charges_acc where we account for plural objects
#     ['b a lion', 'b a a lion lion', 1, (0.5, 0.5), 0.67],               # multi object ===> one predicted for two
    # this example works when using get_single_charges_acc
#     ['b a lion', 'b a a lion lion', 1, (1, 1), 1],               # multi object ===> we neglect the extra color

    ['o b eagle', 'B O lion rampant', 0, (0, 0), 0],    
    ['a g lion rampant', 'O G eagle', 0, (0, 1), 0.33],    
    
    # plural
    ['a g 3 lions rampant', 'a g 3 lions rampant', 1, (1, 1), 1],  
    ['o g 3 lions rampant', 'a g 3 lions rampant', 0, (1, 1), 0.67],    
    ['a v 3 lions rampant', 'a g 3 lions rampant', 1, (1, 0), 0.67],    
    ['o v 3 lions rampant', 'a g 3 lions rampant', 0, (1, 0), 0.33],    
    ['o v 3 lions rampant', 'a g 2 eagles', 0, (0, 0), 0],    
    ['o v 3 lions rampant', 'a g 3 eagles', 0, (0.5, 0), 0.17],    
    ['o v 3 lions rampant', 'a g 3 eagles doubleheaded', 0, (0.33, 0), 0.11],    
    ['o v 2 lions rampant', 'a g lion rampant', 0, (0.5, 0), 0.17],    
]

max_accuracy = [
    { 'input': [[90, 89, 88], 
                [90, 80, 70], 
                [90, 70, 80]], 
     'output': [1, 0, 2], 
     'max_acc': 86.33}
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
            print(predicted, truth, accuracy)
            assert Accuracy(predicted, truth).get() == accuracy
            
    def test_get_max_accuracy(self):
        for item in max_accuracy: 
            max_index, max_acc = Accuracy('b a lion passt guard', 'b a lion passt guard').get_max_accuracy(item['input'])
            assert max_index == item['output']
            assert max_acc == item['max_acc']
    
