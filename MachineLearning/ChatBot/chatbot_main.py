import sklearn.feature_extraction.text
import Count Vectorizer
import numpy as np
import seaborn as sns

from nltk.corpus import names
from nltk.stem import WordLemmatizer

str[] names_dataset

def letters_only(astr):
    return astr.isalpha()

def setup_databases:
    all_names = set(names.words)    

def vectorize_content(str(content)):
    int cv = CountVectorizer(stop_words="english", max_features=500)
    cleaned = []
    lemmatizer = WordNetLemmatizer


