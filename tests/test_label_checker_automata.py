from unittest import TestCase
from src.label_checker_automata import LabelCheckerAutomata

automata = LabelCheckerAutomata()
valid_labels = {
    'O G lion rampant': 'ccom',
    'A G lion rampant': 'ccom',
    'G A fess dancetty': 'ccom',
    'O X GB fess checky': 'ccccom',
    'A GV 3 roses slipped': 'cccnom',
    'O S 6 boars': 'ccno',
    'S AO cross acc. 4 fleurs-de-lis': 'cccopno',
    'G A branch with 3 oak leaves erect': 'ccopnoom',
    'S AO lion chained cr. & border': 'cccommpb',
    "O S eagle's head": 'ccom',
    'B O 3 lions': 'ccno',
    'O G A 3 lions & border checky': 'cccnopbb',
    'O G A lion rampant & border engrailed': 'cccompbb',
    'O GG fess acc. mullet in chf dx': 'cccopoppm',
    'A VG volcano': 'ccco',
    'B O lion guard cr.': 'ccomm',
    'V B lion passt guard': 'ccomm',
    "V V lion's head": 'ccom'
}

invalid_labels = {
    '3 pales; 2 cows; =; = :: label   {OG, OG, B}': 'nn',
    'A G': 'cc',
    'fess G': 'oc',
    'lion': 'o',
    '& lion': 'po',
    'a checky & lion': 'cmpo',
    'Z GO bendy & chief': 'cccmpo',
}

rejected_labels = {
    '3 pales; 2 cows; =; = :: label   {OG, OG, B}': 'nn',
    'safaa coa': 'dd'
}

aligned_parsed_labels = {
    'O G lion rampant': {'colors': ['O','G'], 'objects': ['lion'], 'modifiers': ['rampant'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    'G A eagle doubleheaded': {'colors': ['G','A'], 'objects': ['eagle'], 'modifiers': ['doubleheaded'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    'B G lion passt': {'colors': ['B', 'G'], 'objects': ['lion'], 'modifiers': ['passt'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    'b g lion passt': {'colors': ['b', 'g'], 'objects': ['lion'], 'modifiers': ['passt'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    'b g lion lion': {'colors': ['b', 'g'], 'objects': ['lion', 'lion'], 'modifiers': [], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    'b a lion passt guard': {'colors': ['b', 'a'], 'objects': ['lion'], 'modifiers': ['passt guard'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    'b a g lion passt guard & cross arched & border checky': {'colors': ['b', 'a', 'g'], 'objects': ['lion', 'cross'], 'modifiers': ['passt guard', 'arched'], 'numbers': [], 'positions': ['&', '&'], 'shield_modifiers': ['border','checky']},
    'B O lion guard cr.': {'colors': ['B','O'], 'objects': ['lion'], 'modifiers': ['guard', 'cr.'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    "O O lion's head": {'colors': ['O', 'O'], 'objects': ['lion'], 'modifiers': ['head'], 'numbers': [], 'positions': [], 'shield_modifiers': []},
    "O O 3 lions": {'colors': ['O', 'O'], 'objects': ['lions'], 'modifiers': [], 'numbers': ['3'], 'positions': [], 'shield_modifiers': []},
       
}

class LabelCheckerAutomataTest(TestCase):

    def test_is_valid_passes(self):
        for label, parsed_label in valid_labels.items():
            print('valid label = ', label)
            self.assertTrue(automata.is_valid(label))

    def test_is_valid_fails(self):
        for label, parsed_label in invalid_labels.items():
            print('invalid label = ', label)
            self.assertFalse(automata.is_valid(label))

    def test_parse_label(self):
        for label, parsed_label in valid_labels.items():
            print('accepted label = ', label)
            assert automata.parse_label(label) == parsed_label

        for label, parsed_label in rejected_labels.items():
            print(label)
            with self.assertRaises(ValueError):
                automata.parse_label(label)


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

