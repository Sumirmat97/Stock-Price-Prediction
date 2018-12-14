'''
Generate Word Embeddings
'''

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_news_tokens, generate_missing=False):
    embeddings = clean_news_tokens.apply(lambda x: get_average_word2vec(x, vectors,generate_missing=generate_missing))
    return list(embeddings)

def embeddings(newsAttribute,testNewsAttribute = None):

    '''
        Generates words embeddings of textual data
    :param newsAttribute: Cleaned news training series
    :param testNewsAttribute: Cleaned news testing series
    :return: series of news with word embeddings
    '''

    '''
    tfidfVectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w+')

    newsAttribute = tfidfVectorizer.fit_transform(newsAttribute)
    testNewsAttribute = tfidfVectorizer.transform(testNewsAttribute)
    '''

    tokenizer = RegexpTokenizer(r'\w+')

    newsAttributeTokens = newsAttribute.apply(tokenizer.tokenize)
    if testNewsAttribute:
        testNewsAttributeTokens = testNewsAttribute.apply(tokenizer.tokenize)
    else:
        testNewsAttributeTokens = None

    #downloaded file from https://storage.googleapis.com/kaggle-datasets/5655/8432/GoogleNews-vectors-negative300.bin.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1543479118&Signature=LP%2Beifn4Au%2Bgi0hnF%2BXfMXmmmytJPVBYFYy1x%2FocqTeihhOiaE2ZgNKG1nugN4hEdMnkNP%2FdhHnVsv%2B0hiUoG5MpHA60FS3FTZPpkWgkg50Bb3K6tEZI1bElT%2FzQnk4co4VhXg%2FqrABOkDtlZowwgDf3re6K60wJWTjiKTsjVdzUOsY47RxI%2B%2FpIddcdNVETCNIegpSjoojNafqsq%2BTxxN%2FZkJDYU%2FvXLI9WhlLWAozCpEX5Kc8Wwo9db%2FCNEJORP6qF5eUFtDDIbAYl5ZLuew9B2EuWl%2FS2nG6cBaZbWC64hAoPcl8OI2IbPDtrirQ8FjsBSvAaQSMqlfHuiOPx7Q%3D%3D
    word2vec_path = "../GoogleNews-vectors-negative300.bin"
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


    # Call the functions
    newsAttribute = get_word2vec_embeddings(word2vec, newsAttributeTokens)
    if testNewsAttributeTokens:
        testNewsAttribute = get_word2vec_embeddings(word2vec,testNewsAttributeTokens)

    return newsAttribute,testNewsAttribute
