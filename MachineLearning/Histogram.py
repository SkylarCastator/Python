#Example Code for histogram of highest word counts 
#Python Machine Learning By Example
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

def letters_only(astr):
    return astr.isalpha()

def main():
    cv = CountVectorizer(stop_words="english", max_features=500)
    groups = fetch_20newsgroups()
    cleaned = []
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()

    for post in groups.data:
        cleaned.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))

    transformed = cv.fit_transform(cleaned)
    print (cv.get_feature_names())

    #Set sns to plot data
    sns.distplot(np.log(transformed.toarray().sum(axis=0)))
    plt.xlabel('Log Count')
    plt.ylabel('Frequency')
    plt.title('Distribution Plot of 500 Word Counts')
    plt.show()

main()
