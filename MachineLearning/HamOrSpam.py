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

def letters_only(astr):
    return astr.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
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

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
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
            for label in posterior:
                try:
                    posterior[label] = np.exp(posterior[label] - min_log_posterior)
                except:
                    #if ones log value is excessiverly large, asiign it infinity
                    posterior[label] = float('inf')
            #normalize so that all sums up to 1
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
                posteriors.append(posterior.copy())
            return posteriors

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)

len(X_train), len(Y_train)
len(X_test), len(Y_test)

term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior (label_index)
likelihood = get_likelihood(term_docs_train, label_index, smoothing)

term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_tests, prior, likelihood)

correct =0.0
for pred, actual in zip(posterior, Y_test):
    if actual ==1:
        if pred[1] >= 0.5:
            cprrect += 1
    elif pred[0] > 0.5:
        correct += 1
    print('The accuracy on {0} testing samples is : {1:.1f}%'.format(len(Y_test), correct/len(Y_test)*100))


#Simple Native Bayes System
from sklearn.native_bayes import MultnomialNB

#clf = MultinomialNB(alpha=1.0, fit prior=True)
#clf.fit(term_docs_train, Y_train)
#predict_prob = clf.predict_proba(term_docs_test)
#prediction_prob[0:10]
#accuracy = clf.score(term_docs_test,Y_test)
#print('The accuracy using MutlinoomialNB is: (0:.1f)%'.format(accuracy*100))

#Gets term inverse document frequency within emails
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def inversedocumentfrequency:
    smoothing_factoring_option = [1.0, 2.0, 3.0, 4.0, 5.0]
    auc_record = defaultdict(float)
    for train_indices, test_indices in k_fold.split(cleaned_emails, labels):
      X_train, Xtest = cleaned_emails_np[train_indices], cleaned_emails_np[test_indices]
      Y_train, Y_test = labels_np[train_indices], labels_np[test_indices]
      tfiddf_vectorizer = TfidfVEctorizer(sublinear_tf=True, max_df=0.5, stop_words='english', max_features=8000)
      term_docs_train = tfidf_vectorizer.fit_transform(X_train)
      term_docs_test = tfidf_vectorizer.transform(X_test)
      for smoothing_factor in smoothing_factor_option:
          clf = MultinomialNB(alpha=smoothing_factor, fit_prior=True)
          clf.fit(term_docs_train, Y_train)
          predition_prob = clf.predict_prob[:, 1]
          auc = roc_auc_score(Y_test, pos_prob)
          auc_recording[smoothing_factor] += auc
          print('max features smoothing fit prior auc')
          for smoothing, smoothing_record in auc_record.iteritems():
              print('8000 {0} true {1:.4f}'.format(smoothing, smoothing_record/k))

inversedocumentfrequency()



