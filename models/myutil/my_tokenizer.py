import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

class my_tokenizer:

    def __init__(self, stop_words, lemmatizer):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer

    def my_tokenize(self, text):
        """
        Tokenize the input text by normalizing, removing punctuations,
        lemmatizing and removing stop words
        """
        sents = sent_tokenize(text)
        tokens = []
        for sent in sents:
          # normalize case and remove punctuation
          text = re.sub(r"[^a-zA-Z0-9]", " ", sent.lower())

          # tokenize text
          word_tokens = word_tokenize(text)

          # lemmatize and remove stop words
          sent_tokens = [self.lemmatizer.lemmatize(word, pos='v') for word in word_tokens if word not in self.stop_words]

          # stemming
          sent_tokens = [PorterStemmer().stem(w) for w in sent_tokens]

          tokens = tokens + sent_tokens

        return tokens