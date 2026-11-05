import re
import unicodedata

def clean_text(text):
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters
    text = "".join(
        c for c in text
        if unicodedata.category(c)[0] != "C"
    )

    # Keep ASCII characters and whitespace
    text = re.sub(r"[^\x00-\x7F]", "", text)

    return text

def clean_corpus(corpus):
    return [clean_text(text) for text in corpus]