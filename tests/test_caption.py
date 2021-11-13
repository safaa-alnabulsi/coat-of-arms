from unittest import TestCase
from src.caption import Caption

struc_labels = {
'B A lion': {'shield': {'color': 'B', 'modifiers': []}, 'objects': [{'charge': 'lion', 'color': 'A', 'modifiers': []}]},
    
'S O lion passt': {'shield': {'color': 'S', 'modifiers': []}, 'objects': [{'charge': 'lion', 'color': 'O', 'modifiers': ['passt']}]},
    
'b a lion passt guard': {'shield': {'color': 'b', 'modifiers': []}, 'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': ['passt guard']}]},
    
'b a g lion lion': {'shield': {'color': 'b', 'modifiers': []}, 'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': []}, {'charge': 'lion', 'color': 'g', 'modifiers': []}]},
 
'b a g lion passt guard & cross arched & checky': {'shield': {'color': 'b', 'modifiers': ['checky']}, 'objects': [{'charge': 'lion', 'color': 'a', 'modifiers': ['passt guard']}, {'charge': 'cross', 'color': 'g', 'modifiers': ['arched']}]}


}

class CaptionTest(TestCase):

    def test_get_stuctured(self):
        for label, struc_label in struc_labels.items():
            assert Caption(label).get_structured() == struc_label

