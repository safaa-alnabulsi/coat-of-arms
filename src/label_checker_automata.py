from automata.fa.dfa import DFA
import src.alphabet as alphabet


class LabelCheckerAutomata:
    def __init__(self, colors=alphabet.COLORS,
                 objects=alphabet.OBJECTS,
                 modifiers=alphabet.MODIFIERS,
                 positions=alphabet.POSITIONS,
                 numbers=alphabet.NUMBERS):
        self.dfa = DFA(
            states={'q0', 'q1', 'q2', 'q3', 'q4', 'q5'},
            input_symbols={'c', 'o', 'm', 'p', 'n'},  # c: color, o: object, m: modifier, p: position, n: number
            transitions={
                'q0': {'c': 'q1', 'p': 'q0', 'o': 'q0', 'm': 'q0', 'n': 'q0'},
                'q1': {'c': 'q1', 'o': 'q2', 'm': 'q1', 'p': 'q1', 'n': 'q5'},
                'q5': {'c': 'q5', 'o': 'q2', 'm': 'q5', 'p': 'q5', 'n': 'q5'},
                'q2': {'o': 'q2', 'm': 'q3', 'p': 'q4', 'c': 'q2', 'n': 'q5'},
                'q3': {'m': 'q3', 'o': 'q2', 'p': 'q4', 'c': 'q3', 'n': 'q5'},
                'q4': {'p': 'q4', 'o': 'q2', 'm': 'q3', 'c': 'q4', 'n': 'q4'},
            },
            initial_state='q0',
            final_states={'q2', 'q3'}
        )
        self.objects = objects
        self.colors = colors
        self.modifiers = modifiers
        self.positions = positions
        self.numbers = numbers

    def is_valid(self, label):
        parsed_label = self.parse_label(label)
        return self.dfa.accepts_input(parsed_label)

    def parse_label(self, label):
        chunks = label.split()
        output = ''
        for chunk in chunks:
            if chunk in self.colors:
                output = output + 'c'
            elif chunk in self.objects:
                output = output + 'o'
            elif chunk in self.modifiers:
                output = output + 'm'
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
            'positions': []
        }
        label = label.split(' ')
        for elem, symbol in zip(label, list(parsed_label)):
            if symbol == 'c':
                output['colors'].append(elem)
            elif symbol == 'o':
                output['objects'].append(elem)
            elif symbol == 'm':
                output['modifiers'].append(elem)
            elif symbol == 'n':
                output['numbers'].append(elem)
            elif symbol == 'p':
                output['positions'].append(elem)

        return output
        
    
    @staticmethod
    def get_combination_color(chunk):
        return 'c' * len(chunk)
