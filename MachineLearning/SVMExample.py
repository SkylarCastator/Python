from collections import Counter
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()


def letters_only(astr):
    return astr.isalpha()


def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(
            ' '.join([lemmatizer.lemmatize(word.lower())
                      for word in doc.split()
                      if letters_only(word)
                      and word not in all_names]))
    return cleaned_docs


# Example Project using Support Vector Machine
categories = ['comp.graphics', 'sci.space']
data_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Clean test samples
cleaned_train = clean_text(data_train.data)
label_train = data_train.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target
len(label_train), len(label_test)

Counter(label_train)
Counter(label_test)
tfid_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', max_features=8000)
term_docs_train = tfid_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfid_vectorizer.transform(cleaned_test)

# Sort categories
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(term_docs_train, label_train)
accuracy = svm.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy * 100))

# Multi-class SVM Categorize
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'rec.sport.hockey']
data_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Clean up samples
cleaned_train = clean_text(data_train.data)
label_test = data_test.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target
term_docs_train = tfid_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfid_vectorizer.transform(cleaned_test)

# Sort Categories of objects
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(term_docs_train, label_train)
accuracy = svm.score(term_docs_test, label_test)
print('The accuracy on testing set is {0:1f}%'.format(accuracy * 100))

# Check performace
prediction = svm.predict(term_docs_test)
report = classification_report(label_test, prediction)
print(report)
