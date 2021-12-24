from src.label_checker_automata import LabelCheckerAutomata
from src.armoria_api import ArmoriaAPIPayload
from src.alphabet import SHIELD_MODIFIERS

class Caption:

    # (e.g. "A lion rampant")
    def __init__(self, label, support_plural=False): 
        self.label = label
        self.support_plural=support_plural

    @property
    def is_valid(self):
        simple_automata = LabelCheckerAutomata(support_plural=self.support_plural)
        return simple_automata.is_valid(self.label)

    # (e.g. “com")
    def get_automata_parsed(self): 
        simple_automata = LabelCheckerAutomata(support_plural=self.support_plural)
        return simple_automata.parse_label(label)

    # {'colors': ['b', 'g'], 'objects': ['lion'], 'modifiers': ['passt'], 'numbers': [], 'positions': []}
    def get_aligned(self):
        simple_automata = LabelCheckerAutomata(support_plural=self.support_plural)
        parsed_label = simple_automata.parse_label(self.label)
        print(self.label, parsed_label)
        return simple_automata.align_parsed_label(self.label, parsed_label)

    def get_armoria_payload_dict(self):
        structured_label = self.get_structured()
        return ArmoriaAPIPayload(structured_label).get_armoria_payload()

    def get_structured(self):
                
        charge = {'color': '','object': '','modifiers': []}
        output = {
            'shield': {},
            'objects': []
        }
        
        aligned_label = self.get_aligned()
        
        # Reserve the first color as a shield color; it's a rule
        try:
            shield_color = aligned_label['colors'][0]
            output['shield'] = {'color': shield_color, 
                    'modifiers': []}
            # remove it from list colors
            aligned_label['colors'].pop(0)

        except IndexError:
            print('No shield color found in this label: "{}"'.format(self.label))
            shield_color = 'A' # DEFAULT_SHIELD
            output['shield']['color'] = {'color': shield_color, 
                    'modifiers': []}
            
        
        for item in aligned_label['shield_modifiers']:
            output['shield']['modifiers'].append(item)
    
        # assign each charge to its color
        for charge, color in zip(aligned_label['objects'], aligned_label['colors']):
            output['objects'].append({'charge': charge ,'color': color , 'modifiers': []})
            
        # assigning modifiers to charges 
        #-> for now, I'm assuming that we have one modifier per charge
        try:
            for i, mod in enumerate(aligned_label['modifiers']):               
                if len(output['objects']) > 1:
                    output['objects'][i]['modifiers'].append(mod)
                else: # when one object only, assign all modifiers to it
                    output['objects'][0]['modifiers'].append(mod)
        except IndexError:
            print('Caption Class - exception: ',aligned_label['modifiers'])

        return output

    # (['A', 'lion rampant’])
    def get_tokenized(self):
        pass

    #  [1,3,2,4,2]
    def get_numericalized():
        pass
  