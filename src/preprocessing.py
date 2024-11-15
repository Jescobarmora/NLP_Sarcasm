import language_tool_python
import spacy
from nltk.corpus import stopwords
from spacy.lang.es.stop_words import STOP_WORDS
import re
import nltk

nltk.download('stopwords', quiet=True)

class Preprocessor:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('es')
        self.nlp = spacy.load('es_core_news_lg')
        stop_nltk = stopwords.words('spanish')
        stop_spacy = list(STOP_WORDS)
        self.stop_words = set(stop_nltk + stop_spacy)

    def correct_text(self, text):
        matches = self.tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        return corrected

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in self.stop_words]
        return ' '.join(tokens)

    def process(self, df):
        df = df.copy()
        df['Locución'] = df['Locución'].apply(self.correct_text)
        df['processed_text'] = df['Locución'].apply(self.preprocess_text)
        return df
