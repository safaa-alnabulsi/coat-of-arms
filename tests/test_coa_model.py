


from unittest import TestCase
from src.baseline.coa_model import get_min_max_acc_images

class TestCoaModel(TestCase):
    
    def test_get_min_max_acc_images(self):
        accuracy_test_list = [50,77,30,10,70,9,200,5,44,65,11]
        image_names_list =  ['image_1','image_2','image_3','image_4','image_5','image_6','image_7','image_8','image_9','image_10','image_11']

        lowest_acc, highest_acc = get_min_max_acc_images(accuracy_test_list, image_names_list)
        
        assert lowest_acc == ['image_8', 'image_6', 'image_4', 'image_11', 'image_3']
        assert highest_acc == ['image_7', 'image_2', 'image_5', 'image_10', 'image_1']