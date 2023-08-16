import multiprocessing as mp
import time
from joblib import Parallel, delayed
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
#from tqdm import tqdm

num_words = 10
#num_topics = 50


def output_top_words(weights, vocabulary, length_document):

    # weights_t = np.transpose(weights)
    # chosen_weights = weights_t[:, :length_document]
    # index = np.transpose(np.transpose(np.argsort(chosen_weights))[-num_words:])
    # words = np.flip(vocabulary[index], axis=1)

    index = np.flip(np.argsort(weights)[:, -num_words:], axis=1)
    words = vocabulary[index]
    print(words)

    return words


def count_frequency_1gram(vocab):
    # frequency_1gram = np.zeros(len(vocab))
    frequency_1gram = np.full(len(vocab), 200)  # 200
    with open('pmi/1gms/vocab', 'r', encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            text = tokens[0].lower()
            count = int(tokens[1])
            if text not in vocab:
                continue
            else:
                idx = vocab[text]
                frequency_1gram[idx] = count

    return frequency_1gram


def count_coocurrence_matrix(fileid, vocab):
    print(fileid)
    co_matrix = np.zeros((len(vocab), len(vocab)))
    with open('pmi/5gms/5gm-{}'.format(str(fileid).zfill(4)), 'r', encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            text = tokens[0]
            count = int(tokens[1])
            kept_words = [w for w in text.lower().split() if w in vocab]
            if len(kept_words) > 1:
                for i in range(len(kept_words) - 1):
                    wa_idx = vocab[kept_words[i]]
                    for j in range(i + 1, len(kept_words)):
                        wb_idx = vocab[kept_words[j]]
                        co_matrix[wa_idx][wb_idx] += count

    return co_matrix


def pmi(weights, vocabulary, length_document, num_topics):

    words = output_top_words(weights, vocabulary, length_document)
    vocab = {}
    for line in words:
        for word in line:
            vocab.setdefault(word, len(vocab))

    mask = np.ones((num_topics, int(num_words * (num_words - 1) / 2)))

    final_frequency_1gram = count_frequency_1gram(vocab)

    multiplication_1gram_topic_words = np.ones((num_topics, int(num_words * (num_words - 1) / 2)))
    for i in range(len(words)):
        count = 0
        for j in range(len(words[0]) - 1):
            wa_idx = vocab[words[i][j]]
            for k in range(j + 1, len(words[0])):
                wb_idx = vocab[words[i][k]]
                if final_frequency_1gram[wa_idx] != 0 and final_frequency_1gram[wb_idx] != 0:
                    multiplication_1gram_topic_words[i][count] = final_frequency_1gram[wa_idx] * final_frequency_1gram[
                        wb_idx]
                count += 1

    for i in range(len(multiplication_1gram_topic_words)):
        for j in range(len(multiplication_1gram_topic_words[0])):
            if multiplication_1gram_topic_words[i][j] == 1:
                mask[i][j] = 0

    results_5 = Parallel(n_jobs=mp.cpu_count())(
        delayed(count_coocurrence_matrix)(fileid, vocab) for fileid in range(118))
    final_co_matrix = np.zeros((len(vocab), len(vocab)))
    for r in results_5:
        final_co_matrix += r
    np.fill_diagonal(final_co_matrix, 0)
    frequency_5gram_topic_words = np.full((num_topics, int(num_words * (num_words - 1) / 2)), 1)  # 40
    for i in range(len(words)):
        count = 0
        for j in range(len(words[0]) - 1):
            wa_idx = vocab[words[i][j]]
            for k in range(j + 1, len(words[0])):
                wb_idx = vocab[words[i][k]]
                if final_co_matrix[wa_idx][wb_idx] != 0:
                    frequency_5gram_topic_words[i][count] = final_co_matrix[wa_idx][wb_idx]
                count += 1

    for i in range(len(frequency_5gram_topic_words)):
        for j in range(len(frequency_5gram_topic_words[0])):
            if frequency_5gram_topic_words[i][j] == 1:
                mask[i][j] = 0

    pmi_for_pairs = np.log(frequency_5gram_topic_words) - np.log(multiplication_1gram_topic_words) + np.log(1024908267229)
    npmi_for_pairs = np.divide(np.log(frequency_5gram_topic_words) - np.log(multiplication_1gram_topic_words) + np.log(1024908267229), np.log(1024908267229) - np.log(frequency_5gram_topic_words))


    # print('PMI: %.4f' % np.mean(pmi_for_pairs[np.nonzero(pmi_for_pairs)]))
    print('PMI: %.4f' % np.mean(pmi_for_pairs))
    print('nPMI: %.4f' % np.mean(npmi_for_pairs))
    #print('nPMI: %.4f' % np.mean(np.median(npmi_for_pairs, axis=1)))

    #print('MSCD: %.4f' % mscd(weights, length_document))
    # print('PMI: %.4f' % np.mean(np.median(np.multiply(np.log(frequency_5gram_topic_words) - np.log(multiplication_1gram_topic_words) + np.log(1024908267229), mask), axis=1)))
    # print('PMI: %.4f' % np.median(np.median(np.multiply(np.log(frequency_5gram_topic_words) - np.log(multiplication_1gram_topic_words) + np.log(1024908267229), mask), axis=1)))
    # print('***********')


def mscd(weights, length_document):

    num_hidden = num_topics
    weights_t = np.transpose(weights)
    chosen_weights = weights_t[:, :length_document]
    cos_squared = np.zeros(int(num_hidden * (num_hidden - 1) / 2))
    count = 0
    for i in range(len(chosen_weights) - 1):
        for j in range(i + 1, len(chosen_weights)):
            cos_squared[count] = np.square(
                np.squeeze(cosine_similarity([chosen_weights[i, :]], [chosen_weights[j, :]])))
            count += 1
    mscd = np.sqrt(2 * np.sum(cos_squared) / (num_hidden * (num_hidden - 1)))

    return mscd


#num_words = 10
#num_topics = 30
#topic_word = sys.argv[1]
#topic_word = np.loadtxt(topic_word)
#vocabulary = np.genfromtxt('./cora/dblp/voc.txt', dtype=str)
#pmi(topic_word, vocabulary, len(vocabulary ))