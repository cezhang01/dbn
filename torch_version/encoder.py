import torch
import torch.nn as nn
import numpy as np
import collections


class ConvLayer(nn.Module):

    def __init__(self, dim_in, dim_out):

        super(ConvLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout_keep_prob = 1

        self.generate_modules()

    def generate_modules(self):

        self.linear_layer = nn.ModuleList([nn.Linear(self.dim_in, self.dim_out)])
        self.att_layer = nn.ModuleList([nn.Linear(2 * self.dim_out, 1)])
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)

    def message_passing(self, self_emb, neigh_emb, act, mode):

        if mode == 'train':
            self_emb = self.dropout(self_emb)
            neigh_emb = self.dropout(neigh_emb)
        self_emb = self.linear_layer[0](self_emb)
        neigh_emb = self.linear_layer[0](neigh_emb)

        att = self.att(self_emb, neigh_emb)
        self_emb_agg = self.agg(self_emb, neigh_emb, att, act)

        return self_emb_agg, att

    def att(self, self_emb, neigh_emb):

        n = torch.div(neigh_emb.size(0), self_emb.size(0)).int()

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, n, 1]), neigh_emb.size())
        emb_concat = torch.cat([self_emb, neigh_emb], dim=1)
        att = torch.reshape(self.att_layer[0](emb_concat), [-1, n])
        att = self.softmax(self.sigmoid(att))

        return att

    def agg(self, self_emb, neigh_emb, att, act):

        att = torch.unsqueeze(att, dim=1)
        neigh_emb = torch.reshape(neigh_emb, [att.size(0), att.size(-1), -1])
        neigh_emb_agg = torch.squeeze(torch.matmul(att, neigh_emb))
        self_emb_agg = act(0.5 * self_emb + 0.5 * neigh_emb_agg)

        return self_emb_agg

    def forward(self, self_emb, neigh_emb, act, mode):

        self_emb_agg, att = self.message_passing(self_emb, neigh_emb, act, mode)

        return self_emb_agg, att


class Encoder(nn.Module):

    def __init__(self, args):

        super(Encoder, self).__init__()
        self.parse_args(args)
        self.generate_modules(args)

    def parse_args(self, args):

        self.device = args.device
        self.num_topics = args.num_topics
        self.ptr_word_emb_dim = args.ptr_word_emb_dim
        self.num_words = args.num_words
        self.num_conv_layers = args.num_conv_layers
        self.minibatch_size = args.minibatch_size
        self.dir_prior = args.dir_prior

    def generate_modules(self, args):

        self.conv_dim = [self.num_words] + [128] * (self.num_conv_layers - 1) + [self.num_topics]

        self.conv_layers = nn.ModuleDict()
        for type in ['docs']:
            self.conv_layers[type] = nn.ModuleList()
            for layer_id in range(self.num_conv_layers):
                dim_in, dim_out = self.conv_dim[layer_id], self.conv_dim[layer_id + 1]
                self.conv_layers[type] += nn.ModuleList([ConvLayer(dim_in, dim_out)])

        self.acts = nn.ModuleList([nn.Tanh()] * (self.num_conv_layers - 1)) + nn.ModuleList([nn.Identity()])

    def get_neighbors(self, doc_ids):

        neighbors = collections.defaultdict(list)
        neighbors['doc_ids'] = [doc_ids]
        for layer_id in range(self.num_conv_layers):
            neighbor_doc_ids = np.reshape(self.data.sampled_neighbors[neighbors['doc_ids'][layer_id]], [-1])
            neighbors['doc_ids'].append(neighbor_doc_ids)

        return neighbors

    def get_features(self, doc_ids):

        neighbors = self.get_neighbors(doc_ids)

        features = collections.defaultdict(list)
        for layer_id in range(self.num_conv_layers + 1):
            # feat_docs = torch.FloatTensor(self.data.doc_raw_feat[neighbors['doc_ids'][layer_id]]).to(self.device)
            feat_docs = self.data.generate_bow(neighbors['doc_ids'][layer_id], normalize=True)
            feat_docs = torch.FloatTensor(feat_docs).to(self.device)
            features['docs'].append(feat_docs)

        return features

    def reparameterization(self, alpha):

        self.B = 10
        gamma_dist = torch.distributions.gamma.Gamma(concentration=alpha + float(self.B), rate=1.0)
        gamma = torch.squeeze(gamma_dist.sample())
        epsilon = (torch.sqrt(9. * (alpha + float(self.B)) - 3.) * (torch.pow(gamma / (alpha + float(self.B) - 1. / 3.), 1. / 3.) - 1.)).detach()
        u = torch.rand([self.B, alpha.size(0), self.num_topics]).to(self.device)
        gamma = self.gamma_rejection_sampling_boosted(alpha, epsilon, u)
        doc_topic_dist = torch.div(gamma, torch.sum(gamma, dim=1, keepdim=True))
        doc_topic_dist = torch.reshape(doc_topic_dist, alpha.size())

        return doc_topic_dist

    def gamma_rejection_sampling(self, alpha, epsilon):

        b = alpha - 1. / 3.
        c = 1. / torch.sqrt(9. * b)
        v = 1. + epsilon * c

        return b * (v ** 3)

    def gamma_rejection_sampling_boosted(self, alpha, epsilon, u):

        i = torch.reshape(torch.arange(0, self.B), [-1, 1, 1]).to(self.device)  # [B, 1, 1]
        alpha_vec = torch.reshape(alpha.tile([self.B, 1]), [self.B, -1, self.num_topics]) + i  # [B, minibatch_size, num_topics] + [B, 1, 1]
        u_pow = torch.pow(u, 1. / alpha_vec) + 1e-10
        gamma_tmp = self.gamma_rejection_sampling(alpha + float(self.B), epsilon)
        gamma = torch.prod(u_pow, dim=0) * gamma_tmp

        return gamma

    def graph_conv_encoder(self, doc_ids, mode):

        features = self.get_features(doc_ids)

        z, emb = collections.defaultdict(list), collections.defaultdict(list)
        emb['docs'] = features['docs']
        for layer_id in range(self.num_conv_layers):
            next_emb = collections.defaultdict(list)
            next_emb['docs'] = []
            for hop in range(self.num_conv_layers - layer_id):
                for type in ['docs']:
                    emb_agg, self.att = self.conv_layers[type][layer_id](self_emb=emb[type][hop],
                                                                         neigh_emb=emb[type][hop + 1],
                                                                         act=self.acts[layer_id],
                                                                         mode=mode)
                    next_emb[type].append(emb_agg)
            emb['docs'] = next_emb['docs']

        doc_emb = emb['docs'][0]

        mean = doc_emb
        mean = nn.BatchNorm1d(self.num_topics).to(self.device)(mean)
        alpha = torch.maximum(torch.FloatTensor([1e-12]).to(self.device), torch.log(1. + torch.exp(mean)))
        doc_topic_dist = self.reparameterization(alpha)
        prior = torch.ones([len(doc_ids), self.num_topics], dtype=torch.float32).to(self.device) * self.dir_prior

        kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
        kld -= torch.sum(torch.lgamma(alpha), dim=1)
        kld += torch.sum(torch.lgamma(prior), dim=1)
        minus = alpha - prior
        test = torch.sum(torch.multiply(minus, torch.digamma(alpha) - torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), [len(doc_ids), 1])), dim=1)
        kld += test
        kld = torch.mean(kld)

        return doc_topic_dist, mean, kld

    def forward(self, doc_ids, data, mode='train'):

        self.data = data
        doc_topic_dist, mean, kld = self.graph_conv_encoder(doc_ids, mode)

        return doc_topic_dist, mean, kld, self.att