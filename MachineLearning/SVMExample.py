#Example Project using Support Vector Machine
categories = ['comp.graphics', 'sci.space']
data_train = feth_20newsgroups(subset='train', categories= categories, random_state=42)
sta_test = fetch_20newsgroups(subset='test',categories=categories, random_state=42)
#Clean test samples
cleaned_train - clean_text(data_train.data)
label_train = data_train.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target
len(label_train), len(label_test)
from collections import Counter
Counter(label_train)
Counter(label_test)
tfid_vectorizer = TfidVectorizer(sublinear_tf = True, max_df=0.5, stop_words='english', max_features=8000)
term_docs_train = tfid_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfidf_vectorizer.transform(cleaned_test)
#Sort categories
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(term_docs_train, label_train)
accuracy = svm.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))



#Multiclass SVM Categorizer
categories = ['alt.atheism', 'talk.religon.misc', 'comp.graphics', 'sci.space', 'rec.sport.hocket']
data_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, random_state = 42)
#Clean up samples
cleaned_train = clean_text(data_test.data)
label_test = data_test.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target
term_docs_train =tfid_vectorizer.fit_transform(clean_train)
term_docs_test = tfidf_vectorizer.transform(clean_test)
#Sort CAtegories of objects
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(term_docs_train, label_train)
accuracy = svm.score(term_docs_test, label_test)
print('The accuracy on testing set is {0:1f}%'.format(accuracy*100))
#Check performace
from sklearn.metrics import classification_report
predition = svm.predict(term_docs_test)
report = classification_report(label_test, prediction)
print (report)



