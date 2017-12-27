import urllib.request as req
import json

# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


if __name__ == "__main__":
	url = "https://raw.githubusercontent.com/ugik/notebooks/master/intents.json"
	intents = json.loads(req.urlopen(url).read())
	print(intents)
	words = []
	classes = []
	documents = []
	ignore_words = ['?']

	#loop through each sentence in out intent patterns
	for intent in intents['intents']:
		for pattern in intent['patterns']:
			# tokenize each word in the sentence
			w = nltk.word_tokenize(pattern)
			# add to word list
			words.extend(w)
			# add to documents in our corpus
			documents.append((w,intent['tag']))
			# add to our classes list
			if intent['tag'] not in classes:
				classes.append(intent['tag'])

	# stem and lower each word and remove duplicates
	words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
	words = sorted(list(set(words)))


	#remove duplicates
	classes = sorted(list(set(classes)))

	print (len(documents), "documents")
	print (len(classes), "classes", classes)
	print (len(words), "unique stemmed words", words)

	# create our training data
	training = []
	output = []

	# create empty array for our output
	output_empty = [0] * len(classes)

	# training set - bag of words for each sentence
	for doc in documents:
		# intiialize bag of words
		bag = []
		# list of tokenized words for the pattern
		pattern_words = doc[0]