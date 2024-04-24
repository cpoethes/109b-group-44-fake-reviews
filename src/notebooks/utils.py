import spacy
import random

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from spacy.training import Example 
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL 
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL

import flair
from textattack.augmentation import WordNetAugmenter

import numpy as np
import pandas as pd

import copy
import time
from func_timeout import func_timeout, FunctionTimedOut
import re

from bs4 import BeautifulSoup

# Prepare a string so it is suitable for direct input into our classifiers
#
def clean_text(text):
    if isinstance(text, float):
        text = ""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only alphabets
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [
        word for word in tokens if word.lower() not in stopwords.words("english")
    ]  # Remove stopwords
    if(len(tokens) == 0):
        return ""
    else:
        return " ".join(tokens)

# Partition a string into non-overlapping shingles
# All shingles will be the same size except for [possibly] the last one, which will have length=(len(s) % l)
# Be aware that this function does not perform exactly the same transformations as clean_text().
# If desired, clean_text() should be applied after to_shingles()
#
def to_shingles(s, l=10):
    a = re.sub('[^A-Za-z0-9_]+', ' ', s).lower().strip().split(" ")
    rv = []
    for i in range(0, int((len(a) + (l - 1)) / l)):
        lo = i * l
        hi = lo + l if lo + l <= len(a) else len(a)
        rv = rv + [" ".join(a[lo:hi])]
    return(rv)


# Build and train a spaCy binary classifier
#
def build_classifier(x_, y_, epochs=2):
    
    # Code adapted from Chapter 8 of Mastering spaCy
    classifier = spacy.load("en_core_web_sm") 

    config = { 
       "threshold": 0.5, 
       "model": DEFAULT_MULTI_TEXTCAT_MODEL 
    } 

    textcat = classifier.add_pipe("textcat", config=config) 
    train_examples = []
    for idx, text in enumerate(x_):
        label = {"spam": bool(y_[idx]), "ham": (not bool(y_[idx]))} 
        train_examples.append(Example.from_dict(classifier.make_doc(str(text)), {"cats": label}))

    textcat.add_label("spam")
    textcat.add_label("ham")
    textcat.initialize(lambda: train_examples, nlp=classifier)
    
    # Train the model
    #
    # Note: We observed that the authors of Mastering spaCy used a multi-category classification for a binary variable without explanation, and we
    # found that the multi-category model with a binary output produced better results in testing than the binary model.
    epochs = 2

    with classifier.select_pipes(enable="textcat"): 
      optimizer = classifier.resume_training() 
      for i in range(epochs): 
        random.shuffle(train_examples) 
        for example in train_examples: 
          classifier.update([example], sgd=optimizer)
    
    return classifier


# Run a binary classification using a spaCy binary classifier
#
def classify(classifier, x_):
    y_hat = []
    for test_sample in x_:
        testdoc = classifier(str(test_sample))
        y_hat.append(testdoc.cats['spam']>0.5)
    y_hat = np.array(y_hat).astype(int)
    return(y_hat)
    

# This generates an altered corpus by substituting some proportion of the words with synonyms from WordNet with the TextAttack module.
# For each string in the corpus, it first divides the string into shingles and then uses WordNetAugmenter to substitute some words.
# WordNetAugmenter has an average exeuction time that is proportional to exp(n) where n is the number of words in an input string, and it occasionally
# takes an extremely long time to return even when processing a small number of words. To achieve reasonable runtime, we split each
# string into shingles consisting of a fixed number of words and utilize a timeout.
#
# This sort of algorithm is sometimes referred to as a "spinner."
#
def wordnet_augment(corpus, spin_pct=0.2, shingle_size=10):
    start_time = int(time.time())
    max_i = len(corpus)
    # Preallocate the list to limit the number of copy operations
    augmented = copy.copy(corpus)
    augmenter = WordNetAugmenter(pct_words_to_swap=spin_pct, fast_augment=True)
    print(f"Augmenting corpus with Wordnet at {spin_pct*100}% substitution...")
    for i in range(0, max_i):
        if(i % 200 == 0):
            current_time = time.time()
            print(f'{current_time - start_time} seconds elapsed, estimated {int(float(max_i - i) / (i + 1) * (current_time - start_time + 1))} seconds remaining')
            print(f"{i} of {max_i}", end="\n")
        shingles = to_shingles(corpus[i], shingle_size)
        augmented_shingles = [""] * len(shingles)
        for j in range(len(shingles)):
            try:
                augmented_shingles[j] = func_timeout(1, lambda: augmenter.augment(shingles[j]))
            except FunctionTimedOut:
                print("timeout on shingle", shingles[j])
                augmented_shingles[j] = shingles[j]
        augmented[i] = " ".join([" ".join(a) for a in augmented_shingles])
    print("done.")
    return(augmented)