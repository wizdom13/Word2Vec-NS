import os
import pickle
import matplotlib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot

from arguments import plot_args


def plot(args):
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    words = sorted(wc, key=wc.get, reverse=True)[:args.top_k]

    if args.model == 'pca':
        model = PCA(n_components=2)
    elif args.model == 'tsne':
        model = TSNE(n_components=2, verbose=1, perplexity=30, init='pca', method='exact', n_iter=5000)

    word2idx = pickle.load(open('data/word2idx.dat', 'rb'))
    idx2vec = pickle.load(open('data/idx2vec.dat', 'rb'))

    X = [idx2vec[word2idx[word]] for word in words]
    X = model.fit_transform(X)

    pyplot.figure(figsize=(18, 18))

    for i in range(len(X)):
        pyplot.text(X[i, 0], X[i, 1], words[i], bbox=dict(facecolor='blue', alpha=0.1))

    pyplot.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    pyplot.ylim((np.min(X[:, 1]), np.max(X[:, 1])))

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    pyplot.savefig(os.path.join(args.result_dir, args.model) + '.png')


if __name__ == '__main__':
    plot(plot_args())
