# ==========================================
# Text Preprocessing Module
# CodeAlpha - AI FAQ Chatbot
# ==========================================

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Initialize tools
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and normalize text for FAQ matching.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and lemmatize
    processed_tokens = []

    for token in tokens:

        if token not in stop_words:

            token = lemmatizer.lemmatize(token)

            processed_tokens.append(token)

    # Return processed text
    return " ".join(processed_tokens)
