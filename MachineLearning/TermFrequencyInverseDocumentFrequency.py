from sklearn.feature_extraction.text import TfidfVectorizer
smoothing_factoring_option = [1.0, 2.0, 3.0, 4.0, 5.0]
from collections import defaultdict
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

