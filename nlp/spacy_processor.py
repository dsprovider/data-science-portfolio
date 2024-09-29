import spacy
from .base_nlp_processor import BaseNLPProcessor
# from base_nlp_processor import BaseNLPProcessor

class SpacyProcessor(BaseNLPProcessor):
    def __init__(self, text):
        super().__init__(text)
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = None
    
    def tokenize(self):
        self.doc = self.nlp(self.preprocess())
        return [token.text for token in self.doc]

    def pos_tagging(self):
        return [(token.text, token.pos_) for token in self.doc]

    def named_entity_recognition(self):
        return [(ent.text, ent.label_) for ent in self.doc.ents]
