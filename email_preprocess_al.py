import pickle
import numpy as np
from sklearn import cross_validation
import nltk


def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
	authors_file_handler = open(authors_file, "rb")
	authors = pickle.load(authors_file_handler)
	authors_file_handler.close()

	words_file_handler = open(words_file, "rb")
	word_data = pickle.load(words_file_handler)
	words_file_handler.close()
	### test_size is the percentage of events assigned to the test set
	### (remainder go into training)
	features_train, features_test, labels_train, labels_test =\
		cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


CORPUS = ['This is the first document.',
			'This is the second second document.',
			'And the third one.',
			'Is this the first document?']


def return_token_data(corpus):
	unique_words = []
	all_doc_tokens = []
	
	for this_doc in corpus:
		this_doc_tokens = nltk.tokenize.word_tokenize(this_doc.lower())
		unique_words.extend(this_doc_tokens)
		all_doc_tokens.append(this_doc_tokens)
	
	unique_words = list(set(unique_words))
	return unique_words, all_doc_tokens


def return_tdif_data(corpus):
	unique_words, all_doc_tokens = return_token_data(corpus)
	tdif_array = np.full((len(corpus), len(unique_words)), 0.0)

	for i, doc_tokens in enumerate(all_doc_tokens):
		for j, token in enumerate(unique_words):
			tdif_array[i][j] = doc_tokens.count(token)

	return tdif_array, unique_words, all_doc_tokens
	
print(return_tdif_data(CORPUS))

