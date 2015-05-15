#! /usr/bin/python

__author__="Bhavdeep Sethi <bas2226@columbia.edu>"
__date__ ="$March 11, 2015"

import sys
import cPickle as pickle
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
import time
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

def processFolder():   
	
	categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
	#categories = ['talk.politics.guns', 'talk.politics.mideast','talk.politics.misc', 'talk.religion.misc']
	twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42, categories=categories, remove=('headers', 'footers', 'quotes'))	
	twenty_test = fetch_20newsgroups(subset="test", shuffle=True, random_state=42, categories=categories, remove=('headers', 'footers', 'quotes'))
	docs_test = twenty_test.data
	
	text_clf_nb = Pipeline([('vect', CountVectorizer(stop_words="english", max_features=20000)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words="english", max_features=20000)), ('tfidf', TfidfTransformer()),  ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])	
	text_clf_nb = text_clf_nb.fit(twenty_train.data, twenty_train.target)
	text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)

	predicted_nb = text_clf_nb.predict(docs_test)
	predicted_svm = text_clf_svm.predict(docs_test)

	print "NB: ",str(1.0-numpy.mean(predicted_nb == twenty_test.target))
	print "SVM: ",str(1.0-numpy.mean(predicted_svm == twenty_test.target))

	return #(numpy.array(contentOuter), numpy.array(labelOuter))


if __name__ == "__main__":
	if len(sys.argv)!=1: # Expect exactly one argument: the count data file
		usage()
		sys.exit(2)

	#outputPath = sys.argv[1]       
	start_time = time.time()
	print "Processing Folder"	
	processFolder() 
	#processFolder("test", outputPath+'/20NewsSetOutTest.pkl') 
	#processFolder("test", outputPath+'/20NewsSetOutTest.pkl') 	
	print("--- %s seconds ---" % (time.time() - start_time))    
	'''
	size = len(dataSetOut[0])
	sample = size/10
	train_data_x = dataSetOut[0][0: size-sample]
	train_data_y = dataSetOut[1][0: size-sample]

	valid_data_x = dataSetOut[0][size-sample: size]
	valid_data_y = dataSetOut[1][size-sample: size]

	test_data_x = dataSetOutTest[0]
	test_data_y = dataSetOutTest[1]

	train_data = (train_data_x, train_data_y)
	valid_data = (valid_data_x, valid_data_y)
	test_data = (test_data_x, test_data_y)

	#all_data = (train_data, valid_data, test_data)
	start_time = time.time()
	print "Saving Output"
	dataSetOutFile = open(outputPath+'/20NewsSetOutTrain.pkl', 'wb')
	for curr in train_data:		
		pickle.dump(curr, dataSetOutFile)
	dataSetOutFile.close()

	dataSetOutFile = open(outputPath+'/20NewsSetOutValid.pkl', 'wb')
	for curr in valid_data:		
		pickle.dump(curr, dataSetOutFile)
	dataSetOutFile.close()

	dataSetOutFile = open(outputPath+'/20NewsSetOutTest.pkl', 'wb')
	for curr in test_data:		
		pickle.dump(curr, dataSetOutFile)
	dataSetOutFile.close()

	print("--- %s seconds ---" % (time.time() - start_time))    
	'''
	
	#print str(twenty_train.target_names[twenty_train.target[0]])
	#print str(len(X_train_counts.toarray()[0]))
