#from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
global nltk_wordnet_lemmatizer
nltk_wordnet_lemmatizer = WordNetLemmatizer()
global word_tokenize
word_tokenize = TweetTokenizer()

def text_callback(raw_text):
    if raw_text[0] == '"':
        raw_text = raw_text[1:]
    if raw_text[-1] == '"':
        raw_text = raw_text[:-1]
    #tokens = word_tokenize(raw_text)
    raw_text = raw_text.replace('\n', ' ')
    raw_text = raw_text.replace('\\n', ' ')
    raw_text = raw_text.replace('.\\\\', ' ')
    tokens = word_tokenize.tokenize(raw_text)
    [nltk_wordnet_lemmatizer.lemmatize(item.lower()) for item in tokens]
    return tokens

def label_callback(raw_label):
    return float(raw_label)
