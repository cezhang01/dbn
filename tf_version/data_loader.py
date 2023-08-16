import numpy as np
import collections
from sklearn.preprocessing import normalize
import os


class Data():

    def __init__(self, args):

        print('Preparing data...')
        self.parse_args(args)
        self.load_data()
        self.split_data()
        self.collect_neighbors()
        self.sample_neighbors()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.minibatch_size = args.minibatch_size
        self.training_ratio = args.training_ratio
        self.word_embedding_model = args.word_embedding_model
        self.word_embedding_dimension = args.word_embedding_dimension
        self.structure_embedding_model = args.structure_embedding_model
        self.structure_embedding_dimension = args.structure_embedding_dimension
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.percentage_document_length = args.percentage_document_length

    def load_data(self):

        self.words = self.load_files('../data/' + self.dataset_name + '/contents.txt')
        self.num_docs = len(self.words)
        self.links = self.symmetrize_links(np.loadtxt('../data/' + self.dataset_name + '/links.txt', dtype=int))
        self.num_links = len(self.links)
        self.word_embeddings = np.loadtxt('../data/' + self.dataset_name + '/word_embeddings_' + self.word_embedding_model + '_' + str(self.word_embedding_dimension) + 'd.txt', dtype=float)
        if self.model_name == 'd2bn':
            self.structure_embeddings = np.loadtxt('../data/' + self.dataset_name + '/training_structure_embeddings_' + self.structure_embedding_model + '_' + str(self.structure_embedding_dimension) + 'd.txt', dtype=float)
            if len(self.structure_embeddings) > int(self.num_docs * self.training_ratio):
                self.structure_embeddings = self.structure_embeddings[:int(self.num_docs * self.training_ratio)]
        self.labels_available = os.path.isfile('../data/' + self.dataset_name + '/labels.txt')
        if self.labels_available:
            self.labels = np.loadtxt('../data/' + self.dataset_name + '/labels.txt')
            self.num_labels = len(np.unique(self.labels))
        self.voc = np.genfromtxt('../data/' + self.dataset_name + '/voc.txt', dtype=str)
        self.num_words = len(self.voc)

        self.docs = []
        for doc_id in range(len(self.words)):
            self.docs.append(np.mean(self.word_embeddings[self.words[doc_id]], axis=0))
        self.docs = np.array(self.docs)

    def load_files(self, file_path):

        files = collections.defaultdict(list)
        with open(file_path) as file:
            for row_id, row in enumerate(file):
                row = row.split()
                row = [int(i) for i in row]
                files[row_id] = row

        return files

    def symmetrize_links(self, links):

        symmetric_links = []
        for link in links:
            if link[0] == link[1]:
                continue
            symmetric_links.append([link[0], link[1]])
            symmetric_links.append([link[1], link[0]])
            # symmetric_links.append([link[0], link[0]])
            # symmetric_links.append([link[1], link[1]])
        symmetric_links = np.unique(symmetric_links, axis=0)

        return symmetric_links

    def split_data(self):

        split_idx = int(self.num_docs * self.training_ratio)
        self.training_words, self.test_words, self.training_docs, self.test_docs = collections.defaultdict(list), collections.defaultdict(list), [], []
        for doc_id in range(self.num_docs):
            if doc_id < split_idx:
                self.training_words[doc_id] = self.words[doc_id]
                self.training_docs.append(self.docs[doc_id])
            else:
                self.test_words[doc_id] = self.words[doc_id]
                self.test_docs.append(self.docs[doc_id])
        self.training_docs, self.test_docs = np.array(self.training_docs), np.array(self.test_docs)
        self.num_training_docs, self.num_test_docs = len(self.training_docs), len(self.test_docs)

        if self.labels_available:
            self.training_labels, self.test_labels = self.labels[:split_idx], self.labels[split_idx:]

    def collect_neighbors(self):

        neighbors_set, self.test_links = collections.defaultdict(set), []
        for link in self.links:
            if link[0] < self.num_training_docs and link[1] < self.num_training_docs:
                neighbors_set[link[0]].add(link[1])
                neighbors_set[link[1]].add(link[0])
            elif link[0] >= self.num_training_docs and link[1] < self.num_training_docs:
                neighbors_set[link[0]].add(link[1])
            elif link[0] < self.num_training_docs and link[1] >= self.num_training_docs:
                neighbors_set[link[1]].add(link[0])
            elif link[0] >= self.num_training_docs and link[1] >= self.num_training_docs:
                self.test_links.append([link[0], link[1]])
                self.test_links.append([link[1], link[0]])
        self.neighbors = collections.defaultdict(list)
        for doc_id in neighbors_set:
            self.neighbors[doc_id] = list(neighbors_set[doc_id])
        self.test_links = np.unique(self.test_links, axis=0)
        # for doc_id in range(self.num_docs):
        #     self.neighbors[doc_id].append(doc_id)

    def sample_neighbors(self):

        self.sampled_neighbors = []
        for doc_id in range(self.num_docs):
            if len(self.neighbors[doc_id]) == 0:
                self.sampled_neighbors.append([doc_id] * self.num_sampled_neighbors)
            else:
                neighbors = [doc_id] + self.neighbors[doc_id]
                replace = len(neighbors) < self.num_sampled_neighbors
                neighbor_idx = np.random.choice(len(neighbors), size=self.num_sampled_neighbors, replace=replace)
                self.sampled_neighbors.append([neighbors[idx] for idx in neighbor_idx])
        self.sampled_neighbors = np.array(self.sampled_neighbors)

    def generate_bow(self, doc_ids, normalize=False):

        docs_bow = np.zeros([len(doc_ids), self.num_words], dtype=float)
        for idx, doc_id in enumerate(doc_ids):
            for word_id in self.words[int(doc_id)]:
                docs_bow[idx, word_id] += 1
            if normalize and np.sum(docs_bow[idx]) != 0:
                docs_bow[idx] = docs_bow[idx] / np.sum(docs_bow[idx])

        return docs_bow

    def generate_adj(self, doc_ids, normalize=False):

        adj = np.zeros([len(doc_ids), self.num_training_docs], dtype=float)
        for idx, doc_id in enumerate(doc_ids):
            if doc_id < self.num_training_docs:
                neighbors = np.array([int(doc_id)] + self.neighbors_dict[int(doc_id)])
            else:
                neighbors = np.array(self.neighbors_dict[int(doc_id)])
            if len(neighbors) == 0:
                continue
            adj[idx][neighbors] = 1
            if normalize:
                adj[idx] = adj[idx] / np.sum(adj[idx])

        return adj