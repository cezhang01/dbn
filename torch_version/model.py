import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from decoder import Decoder
import collections


class Model(nn.Module):

    def __init__(self, args, data):

        super(Model, self).__init__()
        self.device = args.device
        self.data = data
        self.parse_args(args)
        self.show_config()
        self.generate_modules(args)

    def parse_args(self, args):

        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.num_topics = args.num_topics
        self.training_ratio = args.training_ratio
        self.num_docs = self.data.num_docs
        self.num_training_docs = self.data.num_training_docs
        self.num_test_docs = self.data.num_test_docs
        self.num_links = self.data.num_links
        self.num_words = self.data.num_words
        args.num_words = self.num_words
        if self.data.labels_available:
            self.num_labels = self.data.num_labels
        self.ptr_word_emb_model = self.data.ptr_word_emb_model
        self.ptr_word_emb_dim = self.data.ptr_word_emb_dim
        args.ptr_word_emb_dim = self.ptr_word_emb_dim
        self.ptr_str_emb_model = self.data.ptr_str_emb_model
        self.ptr_str_emb_dim = self.data.ptr_str_emb_dim
        args.ptr_str_emb_dim = self.ptr_str_emb_dim
        self.num_conv_layers = args.num_conv_layers
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_negative_samples = args.num_negative_samples
        self.num_epochs = args.num_epochs
        self.num_ot_iter = args.num_ot_iter
        self.learning_rate = args.learning_rate
        self.minibatch_size = args.minibatch_size
        self.reg_ot = args.reg_ot
        self.reg_structure = args.reg_structure
        self.reg_kld = args.reg_kld
        self.dir_prior = args.dir_prior
        self.warm_up_epochs = self.num_epochs

    def show_config(self):

        print('******************************************************')
        print('pytorch version:', torch.__version__)
        print('numpy version:', np.__version__)
        print('device:', self.device)
        print('model name:', self.model_name)
        print('dataset name:', self.dataset_name)
        print('#topics:', self.num_topics)
        print('training ratio:', self.training_ratio)
        print('#documents:', self.num_docs)
        print('#training documents:', self.num_training_docs)
        print('#total links:', self.num_links)
        print('#words:', self.num_words)
        if self.data.labels_available:
            print('#labels:', self.num_labels)
        print('#convolutional layers:', self.num_conv_layers)
        print('#sampled neighbors:', self.num_sampled_neighbors)
        print('#epochs:', self.num_epochs)
        print('#ot iterations:', self.num_ot_iter)
        print('learning rate:', self.learning_rate)
        print('minibatch size:', self.minibatch_size)
        print('regularizer for ot:', self.reg_ot)
        print('regularizer for structure:', self.reg_structure)
        print('regularizer for kl divergence:', self.reg_kld)
        print('Dirichlet prior:', self.dir_prior)
        print('******************************************************')

    def generate_modules(self, args):

        self.topic_emb = nn.ModuleDict()
        self.topic_emb['word_space'] = nn.Embedding(self.num_topics, self.ptr_word_emb_dim).to(self.device)
        self.topic_emb['str_space'] = nn.Embedding(self.num_topics, self.ptr_str_emb_dim).to(self.device)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def generate_topic_word_cost_matrix(self):

        ptr_word_emb = torch.FloatTensor(self.data.ptr_word_emb).to(self.device)
        ptr_word_emb_norm = nn.functional.normalize(ptr_word_emb, dim=1)
        topic_emb_norm = nn.functional.normalize(self.topic_emb['word_space'].weight, dim=1)
        self.topic_word_cost_matrix = 1 - torch.matmul(topic_emb_norm, ptr_word_emb_norm.T)  # [num_topics, num_words]

    def generate_topic_str_cost_matrix(self):

        ptr_str_emb = torch.FloatTensor(self.data.ptr_str_emb).to(self.device)
        ptr_str_emb_norm = nn.functional.normalize(ptr_str_emb, dim=1)
        topic_emb_norm = nn.functional.normalize(self.topic_emb['str_space'].weight, dim=1)
        self.topic_str_cost_matrix = 1 - torch.matmul(topic_emb_norm, ptr_str_emb_norm.T)  # [num_topics, num_strs]

    def forward(self, doc_ids, data, mode='train'):

        y_true = collections.defaultdict(list)
        neighbor_doc_ids = np.reshape(self.data.sampled_neighbors[doc_ids], [-1])
        y_true['word'] = torch.FloatTensor(self.data.generate_bow(neighbor_doc_ids, normalize=True)).to(self.device)
        if self.model_name == 'd2bn':
            y_true['adj'] = torch.FloatTensor(self.data.generate_adj(neighbor_doc_ids, normalize=True)).to(self.device)

        loss = 0

        # encoder
        doc_topic_dist, mean, kld, att = self.encoder(doc_ids, data, mode)
        loss += self.reg_kld * kld

        # word decoder
        self.generate_topic_word_cost_matrix()
        recon_loss_word, y_pred = self.decoder(doc_topic_dist, y_true['word'], self.topic_word_cost_matrix, att)
        loss += recon_loss_word

        # structure decoder
        if self.model_name == 'd2bn':
            self.generate_topic_str_cost_matrix()
            recon_loss_adj, _ = self.decoder(doc_topic_dist, y_true['adj'], self.topic_str_cost_matrix, att)
            loss += self.reg_structure * recon_loss_adj

        return loss, mean, y_pred, self.topic_word_cost_matrix