import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from .base_nlp_processor import BaseNLPProcessor
# from base_nlp_processor import BaseNLPProcessor

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class NLTKProcessor(BaseNLPProcessor):
    def __init__(self, text):
        super().__init__(text)
        self.tokens = None
    
    def tokenize(self):
        self.tokens = word_tokenize(self.text)
        return self.tokens

    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))
        self.tokens = [word for word in self.tokens if word.isalnum() and word not in stop_words]
        return self.tokens

    def pos_tagging(self):
        return nltk.pos_tag(self.tokens)

    # def named_entity_recognition(self):
    #     return nltk.ne_chunk(nltk.pos_tag(self.tokens))
    
    def named_entity_recognition(self):
        # NER based on POS tagged tokens
        entities = set()  # Use a set to avoid duplicates
        valid_labels = {"PERSON", "ORGANIZATION", "GPE", "LOCATION", "FACILITY"}  # Valid entity labels to keep
        
        for sent in sent_tokenize(self.text):
            chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)))
            for chunk in chunks:
                if hasattr(chunk, 'label') and chunk.label() in valid_labels:
                    entity = " ".join(c[0] for c in chunk)  # Join the tokens to form the entity
                    entities.add((chunk.label(), entity))  # Add as a tuple (label, entity) to set
        
        return list(entities)  # Convert set back to a list for consistent output