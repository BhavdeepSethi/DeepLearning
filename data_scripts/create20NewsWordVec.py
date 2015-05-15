#! /usr/bin/python

__author__="Bhavdeep Sethi <bas2226@columbia.edu>"
__date__ ="$March 11, 2015"

import sys
import cPickle as pickle
import numpy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import time
import cPickle as pickle
import gensim


def processFolder(type, name):   
	
	categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
	twenty_train = fetch_20newsgroups(subset=type, shuffle=True, random_state=42, categories=categories, remove=('headers', 'footers', 'quotes'))
	

	#count_vect = CountVectorizer(stop_words="english")
	count_vect = CountVectorizer()
	tokenize = count_vect.build_tokenizer()
	sentences = [tokenize(data) for data in twenty_train.data]
	model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=1, workers=4)

	iter = 0
	dataSetOutFile = open(name, 'wb')	
	for sentence in sentences:
		currVec = numpy.zeros((200))
		minVec = numpy.zeros((200))
		maxVec = numpy.zeros((200))
		minVecSum = numpy.inf
		maxVecSum = 0
		for word in sentence:
			if word in model:
				currVecSum = numpy.sum(model[word])
				if currVecSum < minVecSum:
					minVec = model[word]
				if currVecSum > maxVecSum:
					maxVec = model[word]
				currVec = currVec + model[word]/len(sentence)
		currVec = numpy.append(currVec, minVec)
		currVec = numpy.append(currVec, maxVec)

		labelInner = twenty_train.target[iter]
		#print labelInner
		contentLabel = (currVec, labelInner)
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
	processFolder("train", outputPath+'/20NewsSetWordVecTrain.pkl') 
	processFolder("test", outputPath+'/20NewsSetWordVecTest.pkl') 


