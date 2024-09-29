
class BaseNLPProcessor:
    def __init__(self, text):
        # Initialize with text input
        self.text = text

    def preprocess(self):
        # Example of a common text preprocessing step
        # This can be inherited by NLTK and Spacy processors
        return self.text.lower().strip()

    def get_text(self):
        # This could be a simple helper method to retrieve the raw text
        return self.text