import torch
import torch.nn as nn
import numpy as np
import collections


class Decoder(nn.Module):

    def __init__(self, args):

        super(Decoder, self).__init__()
        self.device = args.device
        self.num_negative_samples = args.num_negative_samples
        self.num_topics = args.num_topics
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_ot_iter = args.num_ot_iter
        self.reg_ot = args.reg_ot
        self.gamma = 20

        self.generate_modules()

    def generate_modules(self):

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.functional.binary_cross_entropy

    def sinkhorn_algorithm(self, dist_1, dist_2, cost_matrix):

        psi_1 = torch.ones_like(dist_1).to(self.device) / torch.FloatTensor([dist_1.size(-1)]).to(self.device)
        psi_2 = torch.ones_like(dist_2).to(self.device) / torch.FloatTensor([dist_2.size(-1)]).to(self.device)
        h = torch.exp(- cost_matrix * self.gamma)
        for _ in range(self.num_ot_iter):
            psi_2 = torch.multiply(dist_2, 1 / (torch.matmul(psi_1, h) + 1e-12))
            psi_1 = torch.multiply(dist_1, 1 / (torch.matmul(psi_2, torch.transpose(h, 0, 1)) + 1e-12))
        ot_distance = torch.sum(torch.multiply(torch.matmul(psi_1, torch.multiply(h, cost_matrix)), psi_2), dim=-1)

        return ot_distance

    def sinkhorn_torch(self, a, b, cost_matrix, numItermax=5000, stopThr=.5e-2, cuda=False):

        if cuda:
            u = (torch.ones_like(a) / a.size()[0]).double().cuda()
            v = (torch.ones_like(b)).double().cuda()
        else:
            u = (torch.ones_like(a) / a.size()[0])
            v = (torch.ones_like(b))

        K = torch.exp(-cost_matrix * self.gamma)
        err = 1
        cpt = 0
        while err > stopThr and cpt < numItermax:
            u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))
            cpt += 1
            if cpt % 20 == 1:
                v = torch.div(b, torch.matmul(K.t(), u))
                u = torch.div(a, torch.matmul(K, v))
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, cost_matrix), v)), dim=0)

        return sinkhorn_divergences

    def optimal_transport(self, doc_topic_dist, y_true, cost_matrix, att):

        doc_topic_dist_repeat = torch.reshape(doc_topic_dist.tile([1, self.num_sampled_neighbors]), [-1, self.num_topics])
        y_true_norm = self.softmax(y_true)
        ot_loss = self.sinkhorn_algorithm(doc_topic_dist_repeat, y_true_norm, cost_matrix)
        ot_loss = torch.reshape(ot_loss, [-1, self.num_sampled_neighbors])
        ot_loss = torch.mean(torch.sum(torch.multiply(att, ot_loss), dim=-1))

        return ot_loss

    def decoder(self, doc_topic_dist, y_true, decoder_para, att):

        y_pred = self.softmax(nn.BatchNorm1d(decoder_para.size(-1)).to(self.device)(torch.matmul(doc_topic_dist, decoder_para)))
        y_pred_repeat = torch.reshape(torch.unsqueeze(y_pred, dim=1).tile([1, self.num_sampled_neighbors, 1]), y_true.size())
        # y_true = y_true / torch.sum(y_true, dim=-1, keepdim=True).tile([1, y_true.size(-1)])
        recon_loss = - torch.sum(torch.multiply(y_true, torch.log(y_pred_repeat + 1e-12)), dim=-1)
        recon_loss = torch.reshape(recon_loss, [-1, self.num_sampled_neighbors])
        recon_loss = torch.mean(torch.sum(torch.multiply(att, recon_loss), dim=-1))

        return recon_loss, y_pred

    def forward(self, doc_topic_dist, y_true, cost_matrix, att):

        recon_loss, y_pred = self.decoder(doc_topic_dist, y_true, 2 - cost_matrix, att)
        ot_loss = self.optimal_transport(doc_topic_dist, self.softmax(y_true), cost_matrix, att)
        recon_loss += self.reg_ot * ot_loss

        return recon_loss, y_pred