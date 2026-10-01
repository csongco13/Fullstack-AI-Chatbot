import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

def _ensure_nltk_data():
    for pkg, path in (("punkt","tokenizers/punkt"),("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg,quiet=True)
            

stemmer = PorterStemmer()
_ensure_nltk_data()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """_
    sentence = ["hello","how","are","you"]
    words = ["hi","hello","I","you","bye","thank", "cool"]
    bag = [    0,     1,    0,  1,      0,      0,  0]
    """
    sentence_words = [stem(w) for w in tokenized_sentence]  # turns each word into a sentence in its root form
                                                            # this will help so diff versions of thw same word match
    # Ex: ["Organizing", "organizes", "organized"] -> ["organ","organ","organ"]
    bag = np.zeros(len(all_words), dtype=np.float32)    # initialize bag of words with all zeroes and give a data type as float 
    for idx, w in enumerate(all_words): # this will give index and current word
        if w in sentence_words: # if word in tokenized sentence
            bag[idx]=1.0
    return bag

