from automata.fa.dfa import DFA
import alphabet


class LabelCheckerAutomata:
    def __init__(self):
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

        self.color = alphabet.COLORS
        self.object = alphabet.OBJECTS
        self.modifier = alphabet.MODIFIERS
        self.position = alphabet.POSITIONS
        self.number = alphabet.NUMBERS

    def is_valid(self, label):
        parsed_label = self.parse_label(label)
        return self.dfa.accepts_input(parsed_label)

    def parse_label(self, label):
        chunks = label.split()
        output = ''
        for chunk in chunks:
            if chunk in self.color:
                output = output + 'c'
            elif chunk in self.object:
                output = output + 'o'
            elif chunk in self.modifier:
                output = output + 'm'
            elif chunk in self.position:
                output = output + 'p'
            elif chunk in self.number:
                output = output + 'n'
            elif chunk.endswith("'s"): # possessive pronouns: G A 3 lion's heads
                size = len(chunk)
                trimmed_chun = chunk[:size - 2]
                if trimmed_chun in self.object:
                    output = output + 'o'
            elif chunk.endswith("s"):  # plural s: G A 3 lions
                size = len(chunk)
                trimmed_chun = chunk[:size - 1]
                if trimmed_chun in self.object:
                    output = output + 'o'
            elif self.is_combination_color(chunk):  # to suppot multi-color   'O X GB fess checky' or 'G AGG chief'
                output = output + self._get_combination_color(chunk)
        return output

    def is_combination_color(self, chunck):
        flag = False
        for ch in chunck:
            if ch in self.color:
                flag = True
            else:
                flag = False
                break

        return flag

    def _get_combination_color(self, chunck):
        return 'c' * len(chunck)
