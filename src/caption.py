from src.label_checker_automata import LabelCheckerAutomata
from src.armoria_api import ArmoriaAPIPayload
from src.alphabet import SHIELD_MODIFIERS


class Caption:

    # (e.g. "A lion rampant")
    def __init__(self, label, support_plural=False):
        self.label = label
        self.support_plural = support_plural

    @property
    def is_valid(self):
        automata = LabelCheckerAutomata(
            support_plural=self.support_plural)
        return automata.is_valid(self.label)

    # (e.g. “com")
    def get_automata_parsed(self):
        automata = LabelCheckerAutomata(
            support_plural=self.support_plural)
        return automata.parse_label(label)

    # {'colors': ['b', 'g'], 'objects': ['lion'], 'modifiers': ['passt'], 'numbers': [], 'positions': []}
    def get_aligned(self):
        automata = LabelCheckerAutomata(
            support_plural=self.support_plural)
        parsed_label = automata.parse_label(self.label)
#         print(self.label, parsed_label)
        return automata.align_parsed_label(self.label, parsed_label)

    def get_armoria_payload_dict(self):
        structured_label = self.get_structured()
        return ArmoriaAPIPayload(structured_label).get_armoria_payload()

    @property
    def is_valid_in_armoria(self):
        try: 
            payload = self.get_armoria_payload_dict()
        except ValueError:
#             print(f'Label {self.label} is invalid in Armoria API')
            
            return False
        
        return True
          
    
    def get_structured(self):
        default_shield_color = 'A'  # DEFAULT_SHIELD color
        charge = {'color': '', 'object': '', 'modifiers': []}
        output = {
            'shield': {},
            'objects': []
        }

        aligned_label = self.get_aligned()
        # ------------------------------------------------------------------
        
        # Exceptional use-case: when there is one object and one color, take the color for the object and use the default shield color
        if len(aligned_label['objects']) > 0 and len(aligned_label['colors']) == 1:
            aligned_label['colors'].insert(0, default_shield_color)
        # ------------------------------------------------------------------

        # Reserve the first color as a shield color; it's a rule
        try:
            shield_color = aligned_label['colors'][0]
            output['shield'] = {'color': shield_color, 'modifiers': []}
            # remove it from list colors
            aligned_label['colors'].pop(0)

        except IndexError:
            print(f'No shield color found in this label: "{self.label}"')
            output['shield'] = {'color': default_shield_color, 'modifiers': []}

        for item in aligned_label['shield_modifiers']:
            output['shield']['modifiers'].append(item)

        # ------------------------------------------------------------------

        # assign each charge to its color & initiate charge dict
        for charge, color in zip(aligned_label['objects'], aligned_label['colors']):
            output['objects'].append(
                {'charge': charge, 'color': color, 'modifiers': [], 'number': '1'})

        # ------------------------------------------------------------------

        # assigning modifiers to charges
        # -> for now, I'm assuming that we have one modifier per charge
        try:
            for i, mod in enumerate(aligned_label['modifiers']):
                # # when one object only, assign all modifiers to it
                if mod == '':
                    continue
                elif len(output['objects']) == 1: 
                    output['objects'][0]['modifiers'].append(mod)
                else: 
                     output['objects'][i]['modifiers'].append(mod)

        except IndexError:
            print(
                f"Caption Class - exception in label {self.label}, {aligned_label['modifiers']}")
        # ------------------------------------------------------------------

        # assigning numbers to charges
        if len(aligned_label['numbers']) > 0:
            for i, _ in enumerate(aligned_label['objects']):
                try:
                    output['objects'][i]['number'] = aligned_label['numbers'][i]
                except IndexError:
                    pass
                
        return output

    # (['A', 'lion rampant’])
    def get_tokenized(self):
        pass

    #  [1,3,2,4,2]
    def get_numericalized():
        pass
