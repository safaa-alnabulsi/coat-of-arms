import pandas as pd
import json
import re

class Dataset:
    def __init__(self, labeling_results_json_file):
        self.all_answers = []
        self.labels = []
        self.refined_answers = []
        self.df = pd.read_json(labeling_results_json_file)
        self.possible_answers = self.df.filter(['possible_answers']).values.reshape(1, -1).ravel().tolist()
        for possible_answer in self.possible_answers:
            chunks = possible_answer.split('<|>')
            for c in chunks:
                self.all_answers.append(c)
                self.refined_answers.append(re.sub(r"\b[a-zA-Z]\b", "", c).lstrip().rstrip())
                ws = re.findall(r'\w+', c)
                for w in ws:
                    self.labels.append(w)
