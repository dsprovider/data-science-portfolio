from bs4 import BeautifulSoup

def extract_news_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <article> tags
    article_tags = soup.find_all('article')

    if not article_tags:
        # If no <article> tags found, default to <p> tags
        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    else:
        # Extract text from all <article> elements
        content = ""
        for article in article_tags:
            paragraphs = article.find_all('p')
            article_content = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])
            content += article_content + "\n\n"  # Adding a newline between articles for clarity

    return content
