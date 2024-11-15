import spacy
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        self.nlp = spacy.load('es_core_news_lg')

    def extract_pos_counts(self, text):
        doc = self.nlp(text)
        pos_counts = doc.count_by(spacy.attrs.POS)
        total_tokens = len(doc)
        features = {
            'nouns': pos_counts.get(self.nlp.vocab.strings['NOUN'], 0) / total_tokens,
            'verbs': pos_counts.get(self.nlp.vocab.strings['VERB'], 0) / total_tokens,
            'adjectives': pos_counts.get(self.nlp.vocab.strings['ADJ'], 0) / total_tokens,
            'adverbs': pos_counts.get(self.nlp.vocab.strings['ADV'], 0) / total_tokens,
        }
        return pd.Series(features)

    def transform(self, df):
        pos_features = df['Locución'].apply(self.extract_pos_counts)
        return pos_features.reset_index(drop=True)
