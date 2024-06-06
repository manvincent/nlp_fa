import nltk
import string
from contractions import contraction_dict, convert_contraction

# get list of stopwords in English
stopwords = nltk.corpus.stopwords.words("english")
# create stemmer
stemmer = nltk.stem.PorterStemmer()

def preprocess_text(text): 
    # remove contractions
    text = convert_contraction(text, contraction_dict())
    
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # lowercase the tokens
    tokens = [token.lower() for token in tokens]
    
    
    # remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove dialogue quotes: '' 
    tokens = [token for token in tokens if token not in ["''", '``']]

    # remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords]

    return tokens 
    