from unittest import TestCase
from src.armoria_api import ArmoriaAPIPayload
from src.caption import Caption


pairs_label_payload = {
'B A lion': {'t1': 'azure', 'shield': 'heater', 
             'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'e', 'size': '1.5'}],
             'ordinaries': []},
'B A eagle': {'t1': 'azure', 'shield': 'heater', 
              'charges': [{'charge': 'eagle', 't': 'argent', 'p': 'e', 'size': '1.5'}],
               'ordinaries': []},
'B A eagle doubleheaded': {'t1': 'azure', 
                           'shield': 'heater', 
                           'charges': [{'charge': 'eagleTwoHeards', 't': 'argent', 'p': 'e', 'size': '1.5'}],
                           'ordinaries': []},
 'b a lion passt guard & border': {'t1': 'azure', 
                                            'shield': 'heater', 
                                            'charges': [{'charge': 'lionPassantGuardant',
                                                         't': 'argent', 'p': 'e', 'size': '1.5'}],     
                                            'ordinaries':[{"ordinary":"bordure", "t":"azure"}]},
    
 'b a a lion passt guard & eagle doubleheaded & border': {'t1': 'azure', 
                                            'shield': 'heater', 
                                            'charges': [{'charge': 'lionPassantGuardant',
                                                         't': 'argent', 'p': 'e', 'size': '1.5'},
                                                       {'charge': 'eagleTwoHeards', 
                                                        't': 'argent', 'p': 'e', 'size': '1.5'}],     
                                            'ordinaries':[{"ordinary":"bordure", "t":"azure"}]},

'B A O lion eagle': {'t1': 'azure', 'shield': 'heater', 
             'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'e', 'size': '1.5'},
                        {'charge': 'eagle', 't': 'or', 'p': 'e', 'size': '1.5'}],
             'ordinaries': []},   
}

two_charge_pos = 'kn'

class ArmoriaAPIPayloadTest(TestCase):
    
    
    def test_get_armoria_payload(self):
        for label, payload in pairs_label_payload.items():       
            struc_label = Caption(label).get_structured()
            assert ArmoriaAPIPayload(struc_label).get_armoria_payload() == payload
            
    def test_get_charge_position_single_charge(self):
        struc_label = Caption('B A lion').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_charge_position() == 'e'

    def test_get_charge_position_two_charges_without_modifiers(self):
        struc_label = Caption('B A O lion eagle').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_charge_position() == two_charge_pos

    def test_get_charge_position_two_charges_with_modifiers(self):      
        struc_label = Caption('B A O lion rampant & eagle doubleheaded').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_charge_position() == two_charge_pos
        
    def test_get_charge_position_two_charges_with_modifiers_10(self):              
        struc_label = Caption('B A O lion rampant & eagle').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_charge_position() == two_charge_pos
        
    def test_get_charge_position_two_charges_with_modifiers_01(self):              
        struc_label = Caption('B A O lion passt guard & eagle').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_charge_position() == two_charge_pos
        
    def test_get_charge_position_two_charges_with_modifiers_and_border(self):      
        struc_label = Caption('B A O lion rampant & eagle doubleheaded & border').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_charge_position() == two_charge_pos

            
        