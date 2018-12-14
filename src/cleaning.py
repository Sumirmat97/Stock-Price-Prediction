'''
Text Cleaning
'''

import re
import string

from nltk.corpus import stopwords

def clean(dataFrame):
    '''
        Calls function to clean textual news data, remove and replace unwanted phrases and characters and make words lowercase
    :param dataFrame: news data frame from main
    :return: cleaned news data series and corresponding closing value series

    '''

    text_field = "news"
    predict_field = "close"
    dataFrame[text_field] = dataFrame[text_field].apply(lambda row: cleaner(row))

    return dataFrame[text_field],dataFrame[predict_field]

def cleanForPrediction(dataFrame):
    '''
        Calls function to clean textual data required in case of prediction of certain days using saved models
    :param dataFrame: news from predictDay function
    :return: cleaned news data frame
    '''

    text_field = "news"
    dataFrame = dataFrame[text_field].apply(lambda row: cleaner(row))
    return dataFrame

def cleaner(text):
    '''
        cleans the text string
    :param text: Row of text as received from function clean
    :return: cleaned text string
    '''

    text = text.lower()
    text = text.split()

    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'0,0', '00', text)
    text = re.sub(r'%', ' percent ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'₹', ' rupees ', text)
    text = re.sub(r'\$', ' dollars ', text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    text = re.sub(r' cr ', ' crores ', text)
    text = re.sub(r' us ', ' united states ', text)
    text = re.sub(r' un ', ' united nations ', text)
    text = re.sub(r' uk ', ' united kingdom ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)

    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)

    return text

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

