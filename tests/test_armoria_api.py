from unittest import TestCase
from src.armoria_api import ArmoriaAPIPayload
from src.caption import Caption

two_charge_pos = 'kn'
two_charge_pos_list = ['k', 'n']

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
                                            'ordinaries': [{"ordinary": "bordure", "t": "azure"}]},

    'b a a lion passt guard & eagle doubleheaded & border': {'t1': 'azure',
                                                             'shield': 'heater',
                                                             'charges': [{'charge': 'lionPassantGuardant',
                                                                          't': 'argent', 'p': two_charge_pos[0], 'size': '0.7'},
                                                                         {'charge': 'eagleTwoHeards',
                                                                          't': 'argent', 'p': two_charge_pos[1], 'size': '0.7'}],
                                                             'ordinaries': [{"ordinary": "bordure", "t": "azure"}]},

    'B A O lion eagle': {'t1': 'azure', 'shield': 'heater',
                         'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': two_charge_pos[0], 'size': '0.7'},
                                     {'charge': 'eagle', 't': 'or', 'p': two_charge_pos[1], 'size': '0.7'}],
                         'ordinaries': []},

    'B A 3 lions': {'t1': 'azure', 'shield': 'heater',
                    'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'def', 'size': '0.5'}],
                    'ordinaries': []},

    'B A O 3 lions 3 eagles': {'t1': 'azure', 'shield': 'heater',
                               'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'ken', 'size': '0.5'},
                                           {'charge': 'eagle', 't': 'or', 'p': 'pqa', 'size': '0.5'}],
                               'ordinaries': []},

    'B A O 3 lions 4 eagles': {'t1': 'azure', 'shield': 'heater',
                               'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'ken', 'size': '0.5'},
                                           {'charge': 'eagle', 't': 'or', 'p': 'pqac', 'size': '0.5'}],
                               'ordinaries': []},

    'B A O 3 lions 4 eagles & border': {'t1': 'azure', 'shield': 'heater',
                                        'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'ken', 'size': '0.5'},
                                                    {'charge': 'eagle', 't': 'or', 'p': 'pqac', 'size': '0.5'}],
                                        'ordinaries': [{"ordinary": "bordure", "t": "azure"}]},


    "O O B A 3 lions 4 eagles cross & border": {'t1': 'or', 'shield': 'heater',
                                                'charges': [{'charge': 'lionRampant', 't': 'or', 'p': 'abc', 'size': '0.3'},
                                                            {'charge': 'eagle', 't': 'azure',
                                                            'p': 'dfgz', 'size': '0.3'},
                                                            {'charge': 'crossHummetty', 't': 'argent', 'p': 'i', 'size': '0.3'}],
                                                'ordinaries': [{"ordinary": "bordure", "t": "azure"}]},
    "B A lion's head": {'t1': 'azure', 'shield': 'heater',
                        'charges': [{'charge': 'lionHeadCaboshed', 't': 'argent', 'p': 'e', 'size': '1.5'}],
                        'ordinaries': []},
    "B A A lion's head eagle": {'t1': 'azure', 'shield': 'heater',
                                'charges': [{'charge': 'lionHeadCaboshed', 't': 'argent', 'p': 'k', 'size': '0.7'},
                                            {'charge': 'eagle', 't': 'argent', 'p': 'n', 'size': '0.7'}],
                                'ordinaries': []},
    "A A 10 lions & border": {'t1': 'argent', 'shield': 'heater',
                              'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'ABCDEFGHIJKLe', 'size': '0.18'}],
                              'ordinaries': [{"ordinary": "bordure", "t": "azure"}]},
    "A A A lion eagle doubleheaded & border": {'t1': 'argent',
                                               'shield': 'heater',
                                               'charges': [{'charge': 'lionRampant', 't': 'argent', 'k': 'ABCDEFGHIJKLe', 'size': '0.70'},
                                                           {'charge': 'eagleTwoHeards', 't': 'argent','p': 'n', 'size': '0.70'}],
                                               'ordinaries': [{"ordinary": "bordure", "t": "azure"}]},

}


class ArmoriaAPIPayloadTest(TestCase):

    def test_get_armoria_payload(self):
        for label, payload in pairs_label_payload.items():
            struc_label = Caption(label, support_plural=True).get_structured()
            assert ArmoriaAPIPayload(
                struc_label).get_armoria_payload() == payload

    def test_get_charge_position_single_charge(self):
        struc_label = Caption('B A lion').get_structured()
        assert ArmoriaAPIPayload(struc_label)._get_positions() == ['e']

    def test_get_charge_position_two_charges_without_modifiers(self):
        struc_label = Caption('B A O lion eagle').get_structured()
        assert ArmoriaAPIPayload(
            struc_label)._get_positions() == two_charge_pos_list

    def test_get_charge_position_two_charges_with_modifiers(self):
        struc_label = Caption(
            'B A O lion rampant & eagle doubleheaded').get_structured()
        assert ArmoriaAPIPayload(
            struc_label)._get_positions() == two_charge_pos_list

    def test_get_charge_position_two_charges_with_modifiers_10(self):
        struc_label = Caption('B A O lion rampant & eagle').get_structured()
        assert ArmoriaAPIPayload(
            struc_label)._get_positions() == two_charge_pos_list

    def test_get_charge_position_two_charges_with_modifiers_01(self):
        struc_label = Caption(
            'B A O lion passt guard & eagle').get_structured()
        assert ArmoriaAPIPayload(
            struc_label)._get_positions() == two_charge_pos_list

    def test_get_charge_position_two_charges_with_modifiers_and_border(self):
        struc_label = Caption(
            'B A O lion rampant & eagle doubleheaded & border').get_structured()
        assert ArmoriaAPIPayload(
            struc_label)._get_positions() == two_charge_pos_list
