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


class ArmoriaAPIPayloadTest(TestCase):

    def test_get_armoria_payload(self):
        for label, payload in pairs_label_payload.items():       
            struc_label = Caption(label).get_structured()
            assert ArmoriaAPIPayload(struc_label).get_armoria_payload() == payload

            
            