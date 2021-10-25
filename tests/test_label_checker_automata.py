from unittest import TestCase
from src.label_checker_automata import LabelCheckerAutomata

automata = LabelCheckerAutomata()
accepted_labels = {
    'O G lion rampant': 'ccom',
    'A G lion rampant': 'ccom',
    'G A fess dancetty': 'ccom',
    'O X GB fess checky': 'ccccom',
    'A GV 3 roses slipped': 'cccnom',
    'O S 6 boars': 'ccno',
    'S AO cross acc. 4 fleurs-de-lis': 'cccopno',
    'G A branch with 3 oak leaves erect': 'ccopnoom',
    'S AO lion chained cr. & border': 'cccommpm',
    "O S eagle's head": 'ccoo',
    'B O 3 lions': 'ccno'
}

rejected_labels = {
    '3 pales; 2 cows; =; = :: label   {OG, OG, B}': 'nn',
    'A G': 'cc',
    'fess G': 'oc'
}

aligned_parsed_labels = {
    'O G lion rampant': {'colors': ['O','G'], 'objects': ['lion'], 'modifiers': ['rampant'], 'numbers': [], 'positions': []},
    'G A eagle doubleheaded': {'colors': ['G','A'], 'objects': ['eagle'], 'modifiers': ['doubleheaded'], 'numbers': [], 'positions': []},
    'B G lion passt': {'colors': ['B', 'G'], 'objects': ['lion'], 'modifiers': ['passt'], 'numbers': [], 'positions': []},
    'b g lion passt': {'colors': ['b', 'g'], 'objects': ['lion'], 'modifiers': ['passt'], 'numbers': [], 'positions': []}

}

class LabelCheckerAutomataTest(TestCase):

    def test_is_valid_passes(self):
        for label, parsed_label in accepted_labels.items():
            self.assertTrue(automata.is_valid(label))

    def test_is_valid_fails(self):
        for label, parsed_label in rejected_labels.items():
            self.assertFalse(automata.is_valid(label))

    def test_parse_label(self):
        for label, parsed_label in accepted_labels.items():
            print(label)
            assert automata.parse_label(label) == parsed_label

        for label, parsed_label in rejected_labels.items():
            print(label)
            assert automata.parse_label(label) == parsed_label

    def test_is_combination_color(self):
        assert automata.is_combination_color('AA') == True
        assert automata.is_combination_color('AGO') == True
        assert automata.is_combination_color('&s') == False

    def test_get_combination_color(self):
        print(automata.get_combination_color('AA'))
        assert automata.get_combination_color('AA') == 'cc'
        assert automata.get_combination_color('AGO') == 'ccc'

    def test_align_parsed_label(self):
        for label, ouput in aligned_parsed_labels.items():
            parsed_label = automata.parse_label(label)
            assert automata.align_parsed_label(label, parsed_label) == ouput

