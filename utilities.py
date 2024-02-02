from copy import deepcopy as dcopy
import numpy as np
import pickle

import contextlib
import functools
import time

# Additional necessary functions (from config.py)
class dict2class(object):
    """
    Converts dictionary into class object
    Dict key,value pairs become attributes
    """
    def __init__(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)
            
            
def counterbalance(subID):
    if is_odd(subID):
        sub_cb = 1
    elif not is_odd(subID):
        sub_cb = 2
    return(sub_cb)


def is_odd(num):
   return num % 2 != 0


def minmax(array): 
    return (array - min(array)) / (max(array) -min(array))


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


# ## TFP helper functions
# import tensorflow as tf
# import tensorflow_probability as tfp

# def make_val_and_grad_fn(value_fn):
#   @functools.wraps(value_fn)
#   def val_and_grad(x):
#     return tfp.math.value_and_gradient(value_fn, x)
#   return val_and_grad


# @contextlib.contextmanager
# def timed_execution():
#   t0 = time.time()
#   yield
#   dt = time.time() - t0
#   print('Evaluation took: %f seconds' % dt)


# def np_value(tensor):
#   """Get numpy value out of possibly nested tuple of tensors."""
#   if isinstance(tensor, tuple):
#     return type(tensor)(*(np_value(t) for t in tensor))
#   else:
#     return tensor.numpy()

# def run(optimizer):
#   """Run an optimizer and measure it's evaluation time."""
#   optimizer()  # Warmup.
#   with timed_execution():
#     result = optimizer()
#   return np_value(result)
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from string import punctuation

words = word_tokenize(origin_txt, language='english')

words = [token.lower() for token in tokens] # lowercasing all words
words = [token for token in tokens if not token.isdigit()] # removing any tokens that are only digits 
words = [token for token in tokens if token not in punctuation] # remove punctuations
mystopwords = set(stopwords.words("english"))
words = [token for token in tokens if token not in mystopwords] # removes stopwords
words = [token for token in tokens if len(token) > 3] # remove tokens with fewer than four characters'''
'''
