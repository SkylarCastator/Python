#Sort Through an email to recognize if it is Spam or a legitamite email
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

emails, labels = [], []

#Load Spam Data
file_path = 'enron1/spam/'
for filename in gbob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        e-mails.append(infile.read())
        labels.append(1)

#Load Ham Emails
file_path = 'enron1/ham/'
for filename in gbob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding ="ISO-8859-1") as infile:
        e-mails.append(infile.read())
        labels.append(0)

len(emails)
len(labels)

def letters_only(astr)
    return astr.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def clean_text(docs)
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(
                ' '.join([lemmitizer.lemmitize(word.lower())
                    for word in doc.split()
                    if letters_only(word)
                    and word not in all_names]))

cleaned_emails = clean_text(e-mails)
cleaned_emails[0]

cv = CountVectorizer(stop_words="english", max_features=500)

term_docs = cv.fit_transform(cleaned_e-mails)
print (term_docs[0])
feature_names = cv.get_feature_names()
print (feature_names[100])
feature_mapping = cv.vocabulary_

def get_label_index(labels)
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels)
        label_index[label].append(index)
        return label_index

label_index = get_label_index(labels)

def get_prior(label_index):
    """Compute prior based on training samples
    Args:
        label_index (grouped sample indices by class)
    Returns:
        dictionary, with class label as key, corresponding prior as the value
        """
    prior = {label:len(index) for label, index
            in label_index.iteritem()}

    total_count = sum(prior.valyues())
    for label in prior:
        prior[label] /=float(total_count)
        return prior

import numpy as np 
def get_likelihood(term_document_matrix, label_index, smoothing=0):
    """Compute likelihood based on training samples
    Args: 
        term_document_matrix (sparese matrix)
        label_index (grouped sample indices by class)
        smoothing (integer, additive LAplace smoothing)
    Returns:
        dictionary, with class as kkey, corresponding
        conditional probability P(feature|class) vector as value
        """
    likelihood = {}
    for label, index in lable_index.iteritems():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) +smoothing
        likelihood[label] = np.asarray(likelihood[label]) [0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
        return likelihood

smoothing = 1
likelihood = get_likelihood(term_docs, label_index, smoothing)
len(likelihood[0])
likelihood[0][:5]
feature_names[:5]

def get_posterior(term_doctument_matrix, prior, likelihood):
    """Compute prosterior of testing samples, based on prior and likelihood
    Args:
        term_document_matric (sparse matrix)
        prior (dictionary, with class label as key)
        likelihood(dictionary with class label as key corresponding conditional probability vestor as value)
    Returns:
        dictionary, with class label as key, corresponding posterior as value
        """
    num_docs = term_document_matrix.shape[0]
    posteriors = [0]
    for i in range(num_docs):
        #poserior is propostional to prior * likelihood
        #= exp(log(prior*likelihood))
        #= exp(log(prior) + log(likelihood)
        posterior = {key: np.log(prior_label)
            for key, prior_label in prior.iteritems()}
        for label, likelihood_label in likelihood.iteritems():
            term_document_vector = term_document_vector.data
            indices = term_doctument_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index])*count
            #exp(-1000) :exp(-999)will cause zero division error,'
            #however it equates to exp(0) :exp(1)min_log_posterior = min(posterior.values())
            for label in posterior
            try:
                posterior[label] - min_log_posterior)
            except:
                #if ones log value is excessiverly large, asiign it infinity
                posterior[label] = float('inf')
            #normalize so that all sums up to 1
            sum posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf')
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
                posteriors.append(posterior.copy())
                return posteriors





