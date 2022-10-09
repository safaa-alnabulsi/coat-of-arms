


from unittest import TestCase
from src.baseline.coa_model import get_min_max_acc_images

class TestCoaModel(TestCase):
    
    def test_get_min_max_acc_images(self):
        accuracy_test_list = [50,77,30,10,70,9,200,5,44,65,11]
        image_names_list =  ['image_1','image_2','image_3','image_4','image_5','image_6','image_7','image_8','image_9','image_10','image_11']
        predictions_list =  ['aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk']

        lowest_acc, highest_acc = get_min_max_acc_images(accuracy_test_list, image_names_list, predictions_list)
        
        assert lowest_acc == [{'image_8','hh', 5}, {'image_6','ff', 9}, {'image_4','dd', 10}, {'image_11','kk', 11}, {'image_3','cc', 30}]
        assert highest_acc == [{'image_7','gg', 200}, {'image_2','bb', 77}, {'image_5','ee', 70}, {'image_10','jj', 65}, {'image_1','aa', 50}]
        assert len(lowest_acc) == 5
        assert len(highest_acc) == 5
        
    def test_get_min_max_acc_images_with_same_accuracy(self):
        accuracy_test_list = [50,200,30,10,70,5,200,5,44,65,11]
        image_names_list =  ['image_1','image_2','image_3','image_4','image_5','image_6','image_7','image_8','image_9','image_10','image_11']
        predictions_list =  ['aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk']

        lowest_acc, highest_acc = get_min_max_acc_images(accuracy_test_list, image_names_list, predictions_list)

        assert lowest_acc == [{'image_6','ff', 5}, {'image_8','hh', 5}, {'image_4','dd', 10}, {'image_11','kk', 11}, {'image_3','cc', 30}]
        assert highest_acc == [{'image_7','gg', 200}, {'image_2','bb', 200}, {'image_5','ee', 70}, {'image_10','jj', 65}, {'image_1','aa', 50}]
        assert len(lowest_acc) == 5
        assert len(highest_acc) == 5