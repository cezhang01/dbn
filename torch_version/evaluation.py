from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, normalized_mutual_info_score
import numpy as np


def classification_knn(X_train, X_test, Y_train, Y_test):

    for k in [5, 10, 15, 20]:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, Y_train)
        prediction_label = classifier.predict(X_test)
        print('Micro F1 %d: %.4f' % (k, f1_score(Y_test, prediction_label, average='micro')))
        print('Macro F1 %d: %.4f' % (k, f1_score(Y_test, prediction_label, average='macro')))
        # print('Test accuracy %d: %.4f' % (k, accuracy_score(Y_test, prediction_label)))


def clustering_kmeans(X_train, X_test, Y_train, Y_test):

    prediction_label = KMeans(len(np.unique(Y_train))).fit(X_train).predict(X_test)
    print('NMI: %.4f' % (normalized_mutual_info_score(Y_test, prediction_label)))


def link_prediction_auc(test_embeds, test_links, num_training_docs):

    test_links = np.copy(test_links)
    test_links -= num_training_docs
    auc, count = 0, 0
    for row_idx, row in enumerate(test_embeds):
        y_true = np.zeros(len(test_embeds))
        neighbors = test_links[test_links[:, 0] == row_idx]
        if len(neighbors) == 0:
            continue
        y_true[neighbors[:, 1]] = 1
        y_true = np.delete(y_true, row_idx)
        if np.sum(y_true) == 0:
            continue
        distance = np.sum(np.square(np.delete(test_embeds, row_idx, axis=0) - row), axis=1)
        y_score = - distance
        auc += roc_auc_score(y_true, y_score)
        count += 1
    if count > 0:
        auc /= count
        print('Link prediction AUC: %.4f' % auc)


def perplexity(y_pred, y_true):

    power = - np.sum(np.multiply(np.log(y_pred + 1e-12), y_true)) / (np.sum(y_true) + 1e-12)
    print('Perplexity: %.4f' % power)


num_top_words = 10


def output_top_words(weights, voc):

    index = np.flip(np.argsort(weights)[:, -num_top_words:], axis=1)
    words = voc[index]
    print(words)


def topic_diversity(topic_word_dist):

    top_words_idx = np.flip(np.argsort(topic_word_dist)[:, -num_top_words:], axis=1)
    td = len(np.unique(top_words_idx)) / (len(top_words_idx) * num_top_words)
    print('Topic Diversity: %.4f' % td)