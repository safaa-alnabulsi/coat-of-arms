import spacy
from collections import Counter

class Vocabulary:
    def __init__(self,freq_threshold):
        
        # setting the pre-reserved tokens int to string tokens
        # index to string 
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
#         self.itos = {0:"<PAD>"}
        
        # <PAD>: padding, not part of the sequence represents a token which is the default token
        # <UNK>: unlclear, does not exist in our vocabulary 
        # <SOS>: start of sentence
        # <EOS>: end of sentence
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold

        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        #using spacy for the better text tokenization 
        spacy_eng = spacy.blank("en")

        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]    
