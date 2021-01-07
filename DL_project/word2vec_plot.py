from gensim.models import Word2Vec
model = Word2Vec.load("mini_word2vec.model")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

labels = []
tokens = []

for word in model.wv.vocab:
    tokens.append(model[word])
    labels.append(word)

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(18, 18))
for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(10, 4),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()