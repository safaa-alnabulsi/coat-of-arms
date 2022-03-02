from automata.fa.dfa import DFA
import src.alphabet as alphabet


class LabelCheckerAutomata:
    def __init__(self, colors=alphabet.COLORS,
                 objects=alphabet.OBJECTS,
                 modifiers=alphabet.MODIFIERS,
                 positions=alphabet.POSITIONS,
                 numbers=alphabet.NUMBERS, 
                 shield_modifiers=alphabet.SHIELD_MODIFIERS,
                 support_plural=True):
        
        if support_plural:
            transitions=  {
                    'q0': {'c': 'q1', 'p': 'q0', 'o': 'q0', 'm': 'q0', 'n': 'q0', 'b': 'q0'},
                    'q1': {'c': 'q1', 'o': 'q2', 'm': 'q0', 'p': 'q0', 'n': 'q5', 'b': 'q0'},
                    'q5': {'c': 'q5', 'o': 'q2', 'm': 'q0', 'p': 'q0', 'n': 'q5', 'b': 'q0'},
                    'q2': {'o': 'q2', 'm': 'q3', 'p': 'q4', 'c': 'q0', 'n': 'q5', 'b': 'q0'},
                    'q3': {'m': 'q3', 'o': 'q2', 'p': 'q4', 'c': 'q0', 'n': 'q5', 'b': 'q6'},
                    'q6': {'p': 'q0', 'o': 'q0', 'm': 'q0', 'c': 'q0', 'n': 'q0', 'b': 'q6'},
                    'q4': {'p': 'q4', 'o': 'q2', 'm': 'q3', 'c': 'q0', 'n': 'q4', 'b': 'q6'}
                }
            states = {'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'}
            input_symbols = {'c', 'o', 'm', 'p', 'n', 'b'}
        else: # no numbers
            transitions = {
                    'q0': {'c': 'q1', 'p': 'q0', 'o': 'q0', 'm': 'q0', 'b': 'q0'},
                    'q1': {'c': 'q1', 'o': 'q2', 'm': 'q0', 'p': 'q0', 'b': 'q0'},
                    'q2': {'o': 'q2', 'm': 'q3', 'p': 'q4', 'c': 'q0', 'b': 'q0'},
                    'q3': {'m': 'q3', 'o': 'q2', 'p': 'q4', 'c': 'q0', 'b': 'q3'},
                    'q4': {'p': 'q4', 'o': 'q2', 'm': 'q3', 'c': 'q0', 'b': 'q6'},
                    'q6': {'p': 'q0', 'o': 'q0', 'm': 'q0', 'c': 'q0', 'b': 'q6'}
            }
            states = {'q0', 'q1', 'q2', 'q3', 'q4', 'q6'}
            input_symbols = {'c', 'o', 'm', 'p', 'b'}
        
        self.dfa = DFA(
            states=states,
            input_symbols=input_symbols,  # c: color, o: object, m: modifier, p: position, n: number
            transitions=transitions,
            initial_state='q0',
            final_states={'q2', 'q3', 'q6'}
        )
        self.objects = objects
        self.colors = colors
        self.modifiers = modifiers
        self.positions = positions
        self.numbers = numbers
        self.shield_modifiers = shield_modifiers

    def is_valid(self, label):
        try:
            parsed_label = self.parse_label(label)
            return self.dfa.accepts_input(parsed_label)
        except ValueError:
            return False
        
    
    def parse_label(self, label):
        chunks = label.split()
        output = ''
        has_border = False
        for chunk in chunks:
            if chunk == 'border':
                has_border = True
            if chunk.upper() in self.colors:
                output = output + 'c'
            elif chunk in self.objects:
                output = output + 'o'
            elif chunk in self.modifiers:
                if has_border: # there are shared modifiers between charges & shield
                    output = output + 'b'
                else:
                    output = output + 'm'
            elif chunk in self.shield_modifiers:
                output = output + 'b'
            elif chunk in self.positions:
                output = output + 'p'
            elif chunk in self.numbers:
                output = output + 'n'
            elif chunk.endswith("'s"):  # possessive pronouns: G A 3 lion's heads
                size = len(chunk)
                trimmed_chunk = chunk[:size - 2]
                if trimmed_chunk in self.objects:
                    output = output + 'o'
            elif chunk.endswith("s"):  # plural s: G A 3 lions
                size = len(chunk)
                trimmed_chunk = chunk[:size - 1]
                if trimmed_chunk in self.objects:
                    output = output + 'o'
            elif self.is_combination_color(chunk):  # to support multi-color 'O X GB fess checky' or 'G AGG chief'
                output = output + self.get_combination_color(chunk)
            else:
                raise ValueError(f'label "{label}" cannot be parsed. The chunk "{chunk}" cannot be fit into any category.')
        
        return output

    def is_combination_color(self, chunk):
        flag = False
        for ch in chunk:
            if ch in self.colors:
                flag = True
            else:
                flag = False
                break

        return flag

    
    def align_parsed_label(self, label, parsed_label):
        output = {
            'colors': [],
            'objects': [],
            'modifiers': [],
            'numbers': [],
            'positions': [],
            'shield_modifiers': [],
        }
        
        label_ls = label.split(' ')
        
        has_passt_guard = False
        if "passt" in label_ls and "guard" in label_ls:
            has_passt_guard = True
        
        for elem, symbol in zip(label_ls, list(parsed_label)):
            if symbol == 'c':
                output['colors'].append(elem)
            elif symbol == 'o':
                # if elem.endswith("'s"):
                    # size = len(elem)
                    # elem = elem[:size - 2]
                output['objects'].append(elem)
            elif symbol == 'b':
                output['shield_modifiers'].append(elem)
            elif symbol == 'm':
                output['modifiers'].append(elem)
            elif symbol == 'n':
                output['numbers'].append(elem)
            elif symbol == 'p':
                output['positions'].append(elem)
        
        # One exceptional use-case, "passt guard" are treated as one modifier to fix accuracy
        if has_passt_guard: 
            index = output['modifiers'].index('passt')
            output['modifiers'].insert(index, "passt guard")
            output['modifiers'].remove('guard')
            output['modifiers'].remove('passt')

            
        return output
    
    def get_valid_labels(self, all_labels):
        validated_labels = {}
        for label in all_labels:
            if self.is_valid(label):
                validated_labels[label] = 1
            else: 
                validated_labels[label] = 0

        return validated_labels
    

    def get_valid_labels_of(self, all_labels, charge):
        validated_labels = self.get_valid_labels(all_labels)
        charge_labels = []
        for l, valid in validated_labels.items():
            if valid and charge in l: 
                charge_labels.append(l)
        
        return charge_labels

    def get_valid_plural_labels(self, all_labels):
        validated_labels = self.get_valid_labels(all_labels)
        pl_labels = []
        for l, valid in validated_labels.items():
            if valid and self.has_numbers(l): 
                pl_labels.append(l)
        
        return pl_labels
    
    def has_numbers(self, inputString):
         return any(char.isdigit() for char in inputString)

    @staticmethod
    def get_combination_color(chunk):
        return 'c' * len(chunk)
