import pickle
import numpy as np
from sklearn import cross_validation
import nltk


#http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
#td-idf = tf(t,d) * idf(t)

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


def return_tf_idf_data(corpus):
	unique_words, all_doc_tokens = return_token_data(corpus)
	tdif_array = np.full((len(corpus), len(unique_words)), 0.0)
	idf_array = np.full(len(unique_words), 0.0)

	for j, token in enumerate(unique_words):
		
		document_frequency = 0
		for i, doc_tokens in enumerate(all_doc_tokens):
			
			if token in doc_tokens:
				tdif_array[i][j] = doc_tokens.count(token)
				document_frequency+=1;
		idf_array[j] = np.log((1+len(corpus))/(1+document_frequency)) + 1


	tdif_array = tdif_array * idf_array
	return tdif_array, unique_words, all_doc_tokens
	


td_idf_data = return_tf_idf_data(CORPUS)
print(td_idf_data)
print(CORPUS)

