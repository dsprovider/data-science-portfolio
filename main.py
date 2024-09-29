import os
import pprint

from nlp.nltk_processor import NLTKProcessor
from nlp.spacy_processor import SpacyProcessor
from utils.file_loader import load_local_html
from utils.html_parser import extract_news_content

def process_article_nltk(text):
    nltk_processor = NLTKProcessor(text)
    tokens = nltk_processor.tokenize()
    clean_tokens = nltk_processor.remove_stopwords()
    pos_tags = nltk_processor.pos_tagging()
    entities = nltk_processor.named_entity_recognition()
    
    return {
        # "tokens": tokens,
        # "clean_tokens": clean_tokens,
        # "pos_tags": pos_tags,
        "entities": entities
    }

# def process_article_spacy(text):
#     spacy_processor = SpacyProcessor(text)
#     tokens = spacy_processor.tokenize()
#     pos_tags = spacy_processor.pos_tagging()
#     entities = spacy_processor.named_entity_recognition()
    
#     return {
#         "tokens": tokens,
#         "pos_tags": pos_tags,
#         "entities": entities
#     }

def process_news_text():

    news_directory = os.path.join(os.getcwd(), "data")

    for root, dirs, files in os.walk(news_directory):
        for file in files:
            file_path = os.path.join(root, file)
            news_html = load_local_html(file_path) # Read HTML file
            if news_html:
                news_content = extract_news_content(news_html) # Extract news content

                result_nltk = process_article_nltk(news_content)
                pprint.pprint(result_nltk)
                print("====================================================================================\n")

if __name__ == "__main__":
    process_news_text()
