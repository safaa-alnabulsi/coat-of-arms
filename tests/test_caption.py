from unittest import TestCase
from src.caption import Caption

struc_labels = {
    'B A lion': {'shield': {'color': 'B', 'modifiers': []},
                 'objects': [{'charge': 'lion', 'color': 'A', 'modifiers': [], 'number': '1'}]},

    'S O lion passt': {'shield': {'color': 'S', 'modifiers': []},
                       'objects': [{'charge': 'lion', 'color': 'O', 'modifiers': ['passt'], 'number': '1'}]},

    'b a lion passt guard': {'shield': {'color': 'b', 'modifiers': []},
                             'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': ['passt guard'], 'number': '1'}]},

    'b a g lion lion': {'shield': {'color': 'b', 'modifiers': []},
                        'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': [], 'number': '1'},
                                    {'charge': 'lion', 'color': 'g', 'modifiers': [], 'number': '1'}]},

    'b a g lion passt guard & cross arched & border checky': {'shield': {'color': 'b', 'modifiers': ['border', 'checky']},
                                                              'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': ['passt guard'], 'number': '1'},
                                                                          {'charge': 'cross', 'color': 'g', 'modifiers': ['arched'], 'number': '1'}]},

    'b a g lion passt guard & border': {'shield': {'color': 'b', 'modifiers': ['border']},
                                        'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': ['passt guard'], 'number': '1'}]},

    'B O lion guard cr.': {'shield': {'color': 'B', 'modifiers': []},
                           'objects': [{'charge': 'lion', 'color': 'O', 'modifiers': ['guard', 'cr.'], 'number': '1'}]},

    'B O lion passt guard cr.': {'shield': {'color': 'B', 'modifiers': []},
                                 'objects': [{'charge': 'lion', 'color': 'O', 'modifiers': ['passt guard', 'cr.'], 'number': '1'}]},

    'A S bear rampant chained': {'shield': {'color': 'A', 'modifiers': []},
                                 'objects': [{'charge': 'bear', 'color': 'S', 'modifiers': ['rampant', 'chained'], 'number': '1'}]},

    "O O lion's head":  {'shield': {'color': 'O', 'modifiers': []},
                         'objects': [{'charge': 'lion', 'color': 'O', 'modifiers': ['head'], 'number': '1'}]},

    "O O 3 lions":  {'shield': {'color': 'O', 'modifiers': []},
                     'objects': [{'charge': 'lions', 'color': 'O', 'modifiers': [], 'number': '3'}]},

    "O O B 3 lions 4 eagles":  {'shield': {'color': 'O', 'modifiers': []},
                                'objects': [{'charge': 'lions', 'color': 'O', 'modifiers': [], 'number': '3'},
                                            {'charge': 'eagles', 'color': 'B', 'modifiers': [], 'number': '4'}]},

    "O O B 3 lions 4 eagles & border":  {'shield': {'color': 'O', 'modifiers': ['border']},
                                         'objects': [{'charge': 'lions', 'color': 'O', 'modifiers': [], 'number': '3'},
                                                     {'charge': 'eagles', 'color': 'B', 'modifiers': [], 'number': '4'}]},

    "O O B A 3 lions 4 eagles cross & border":  {'shield': {'color': 'O', 'modifiers': ['border']},
                                                 'objects': [{'charge': 'lions', 'color': 'O', 'modifiers': [], 'number': '3'},
                                                             {'charge': 'eagles', 'color': 'B',
                                                              'modifiers': [], 'number': '4'},
                                                             {'charge': 'cross', 'color': 'A', 'modifiers': [], 'number': '1'}]},

}


class CaptionTest(TestCase):

    def test_get_stuctured(self):
        for label, struc_label in struc_labels.items():
            assert Caption(label).get_structured() == struc_label
