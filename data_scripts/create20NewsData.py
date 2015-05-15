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
import cPickle as pickle

def processFolder(type, name):   
	
	categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
	twenty_train = fetch_20newsgroups(subset=type, shuffle=True, random_state=42, categories=categories, remove=('headers', 'footers', 'quotes'))
	count_vect = CountVectorizer(stop_words="english", max_features=25000)
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	iter = 0
	contentOuter = []
	labelOuter = []
	bowArray = X_train_tfidf.toarray()
	dataSetOutFile = open(name, 'wb')
	for data in twenty_train.data:
		contentInner = bowArray[iter]
		labelInner = twenty_train.target[iter]
		#print labelInner
		contentLabel = (contentInner, labelInner)
		pickle.dump(contentLabel, dataSetOutFile)
		iter += 1
	dataSetOutFile.close()
	return #(numpy.array(contentOuter), numpy.array(labelOuter))


if __name__ == "__main__":
	if len(sys.argv)!=2: # Expect exactly one argument: the count data file
		usage()
		sys.exit(2)

	outputPath = sys.argv[1]       
	start_time = time.time()
	print "Processing Folder"	
	processFolder("train", outputPath+'/20NewsSetOutTrain.pkl') 
	processFolder("test", outputPath+'/20NewsSetOutTest.pkl') 
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
