from automata.fa.dfa import DFA
from src.alphabet import Alphabet


class LabelCheckerAutomata:
    def __init__(self):
        self.dfa = DFA(
            states={'q0', 'q1', 'q2', 'q3', 'q4'},
            input_symbols={'c', 'o', 'm', 'p'},  # c: color, o: object, m: modifier, p: position
            transitions={
                'q0': {'c': 'q1', 'p': 'q0', 'o': 'q0', 'm': 'q0'},
                'q1': {'c': 'q1', 'o': 'q2', 'm': 'q1', 'p': 'q1'},
                'q2': {'o': 'q2', 'm': 'q3', 'p': 'q4', 'c': 'q2'},
                'q3': {'m': 'q3', 'o': 'q2', 'p': 'q4', 'c': 'q3'},
                'q4': {'p': 'q4', 'o': 'q2', 'm': 'q3', 'c': 'q4'},
            },
            initial_state='q0',
            final_states={'q2', 'q3'}
        )
        alphabet = Alphabet()
        self.color = alphabet.color
        self.object = alphabet.object
        self.modifier = alphabet.modifier
        self.position = alphabet.position

    def is_valid(self, label):
        parsed_label = self._parse_label(label)
        return self.dfa.accepts_input(parsed_label)

    def _parse_label(self, label):
        chunks = label.split()
        output = ''
        for chunck in chunks:
            if chunck in self.color:
                output = output + 'c'
            elif chunck in self.object:
                output = output + 'o'
            elif chunck in self.modifier:
                output = output + 'm'
            elif chunck in self.position:
                output = output + 'p'
        return output