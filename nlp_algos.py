"""
nlp_algos
~~~~~~~~~
"""
import re
import nltk
import math
import string
import numpy as np
from collections import Counter
from pprint import pprint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel


print(Counter(word_tokenize("""The cat is in the box. The cat likes the box.
                                The box is over the cat.""")))

my_documents = ['The movie was about a spaceship and aliens.'
            ,'I really liked the movie'
            ,'Awesome action scenes, but boring characters.'
            ,'The movie was awful! I hate alien films. I hate this movie! So much! !./movie...!'
            ,'Space is cool! I liked the movie.'
            ,'More space films, please!'
            ,]

my_documents_two = ["Ben studies about computers in Computer Lab.",
                    "Steve teaches at Brown University.",
                    "Data Scientists work on large datasets."]

def term_frequency(term: str, doc: str) -> float:
    """Calculates the term frequency of a given word.
    tf(t, d) = N(t, d) / ||D||
    """
    term = len([s.lower()
                for s in doc.translate(
                    str.maketrans('', '', string.punctuation)).split()
                    if  s == term])
    return term

def inverse_term_frequency(term: str, documents: list) -> float:
    """Calculates the inverse document frequency of a given term
    """
    # step 1: calculate the number of times the current term appears within the whole text
    # term_count = len([s.lower() for doc in documents for s in doc.translate(
    #                 str.maketrans('', '', string.punctuation)).split()
    #                 if s.lower() == term])
    #n_terms = sum([len(doc.split()) for doc in documents])

    # step 2: calculate the number of documents that the term appears in
    total_docs_with_term = 0
    for doc in documents:
        current = doc.translate(str.maketrans('', '', string.punctuation)).lower().split()
        if term in current:
            total_docs_with_term += 1

    if total_docs_with_term > 0:
        # step 3: calculate the idf value
        idf = np.log(float(len(documents)) / total_docs_with_term)
        return idf
    else:
        return 0.0

def tf_idf(documents: list) -> list:
    """Term frequency, inverse term frequency algorithm"""
    unique_words = list(set([s.lower()
                             for doc in documents
                             for s in doc.translate(
                                str.maketrans('', '', string.punctuation)).split()]))
    print(type(unique_words))
    tf_idf_score = {unique_words[i]: term_frequency(unique_words[i], d)*inverse_term_frequency(unique_words[i], documents)
                                 for i, _  in enumerate(unique_words) for d in documents}
    pprint(tf_idf_score)

    return unique_words


tokenized_docs = [word_tokenize(doc.lower()) for doc in my_documents]
dictionary = Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

pprint(term_frequency('movie', my_documents[3]))

pprint(inverse_term_frequency(term='computer', documents=my_documents_two))

pprint(tf_idf(my_documents_two))
