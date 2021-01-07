import os
import sys
import re
from gensim.models import Word2Vec
# from gensim.models.phrases import Phraser, Phrases
TEXT_DATA_DIR = './uza/'

texts = []         # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []        # list of label ids
label_text = []    # list of label texts
# Go through each directory
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            # News groups posts are named as numbers, with no extensions.
            fpath = os.path.join(path, fname)
            f = open(fpath, encoding='utf-8')
            t = f.read()
            texts.append(t)
            f.close()
            labels.append(label_id)
            label_text.append(name)


# Cleaning data - remove punctuation from every newsgroup text
sentences = []
# Go through each text in turn
for ii in range(len(texts)):
    sentences = [re.sub(pattern=r'[\!"“”#$%&\*+,-./:;<=>?@^_`()|~=]',
                        repl='',
                        string=x
                       ).strip().split(' ') for x in texts[ii].split('\n')
                      if not x.endswith('writes:')]
    sentences = [x for x in sentences if x != ['']]
    texts[ii] = sentences



# concatenate all sentences from all texts into a single list of sentences
all_sentences = []
for text in texts:
    all_sentences += text


model = Word2Vec(all_sentences,
                 min_count=3,   # Ignore words that appear less than this
                 size=200,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus
model.save("uza_word2vec.model")