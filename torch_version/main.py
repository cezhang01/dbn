import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from data_loader import *
from model import Model
import os
import time
from tqdm import tqdm
from evaluation import *


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, default='dbn', choices=['dbn', 'd2bn'])
    parser.add_argument('-ne', '--num_epochs', type=int, default=100)  # small datasets == 100, large datasets == 5 or 10
    parser.add_argument('-no', '--num_ot_iter', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=64)
    parser.add_argument('-dn', '--dataset_name', type=str, default='ml')
    parser.add_argument('-nt', '--num_topics', type=int, default=64)
    parser.add_argument('-reg_ot', '--reg_ot', type=float, default=3)
    parser.add_argument('-reg_str', '--reg_structure', type=float, default=1)
    parser.add_argument('-reg_kld', '--reg_kld', type=float, default=0.001)
    parser.add_argument('-dp', '--dir_prior', type=float, default=1)
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8)
    parser.add_argument('-nl', '--num_conv_layers', type=int, default=2)
    parser.add_argument('-nn', '--num_sampled_neighbors', type=int, default=5)
    parser.add_argument('-neg', '--num_negative_samples', type=int, default=5)
    parser.add_argument('-rs', '--random_seed', type=int, default=519)
    parser.add_argument('-gpu', '--gpu', type=int, default=0)

    return parser.parse_args()


def train(args):

    args.device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'

    print('Preparing data...')
    data_center = DataCenter(args)
    training_data = Data(args, 'train', data_center)
    test_data = Data(args, 'test', data_center)
    training_loader = DataLoader(dataset=training_data, batch_size=args.minibatch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=args.minibatch_size, shuffle=False)

    print('Start training...')
    model = Model(args, data_center).to(args.device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    num_minibatches = len(training_loader)
    t = time.time()
    for epoch_idx in tqdm(range(1, args.num_epochs + 1)):
        one_epoch_loss = 0.0
        # training
        model.train()
        data_center.sample_neighbors()
        for idx, batch in enumerate(training_loader):
            links, doc_ids_neg = batch
            doc_ids_neg = np.reshape(doc_ids_neg, [-1])
            optimizer.zero_grad()
            loss_1, _, _, _ = model(links[:, 0], data_center, mode='train')
            loss_2, _, _, _ = model(links[:, 1], data_center, mode='train')
            # loss_neg, _, _ = model(doc_ids_neg)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                one_epoch_loss += loss.item()
        # testing
        if epoch_idx % 5 == 0 or epoch_idx == 1:
            print('******************************************************')
            print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_idx, args.num_epochs), '\tLoss: %f' % one_epoch_loss)

            model.eval()
            doc_topic_dist, y_pred, topic_word_cost_matrix = [], [], []
            for idx, batch in enumerate(test_loader):
                links, _ = batch
                _, doc_topic_dist_tmp, y_pred_tmp, topic_word_cost_matrix = model(links[:, 0], data_center, mode='test')
                doc_topic_dist_tmp = doc_topic_dist_tmp.detach().cpu().numpy().tolist()
                y_pred_tmp = y_pred_tmp.detach().cpu().numpy().tolist()
                doc_topic_dist.extend(doc_topic_dist_tmp)
                y_pred.extend(y_pred_tmp)
            doc_topic_dist = np.array(doc_topic_dist)
            training_doc_topic_dist = doc_topic_dist[:data_center.num_training_docs]
            test_doc_topic_dist = doc_topic_dist[data_center.num_training_docs:]
            test_y_pred = np.array(y_pred[data_center.num_training_docs:])

            if data_center.labels_available:
                classification_knn(training_doc_topic_dist, test_doc_topic_dist, data_center.training_labels, data_center.test_labels)
                clustering_kmeans(training_doc_topic_dist, test_doc_topic_dist, data_center.training_labels, data_center.test_labels)
            link_prediction_auc(test_doc_topic_dist, data_center.test_links, data_center.num_training_docs)

            topic_word_dist = 2 - topic_word_cost_matrix.detach().cpu().numpy()
            topic_diversity(topic_word_dist)

    output_top_words(topic_word_dist, data_center.voc)
    # save model outputs
    folder = os.path.exists('../data/' + args.dataset_name + '/results')
    if not folder:
        os.makedirs('../data/' + args.dataset_name + '/results')
    np.savetxt('../data/' + args.dataset_name + '/results/doc_topic_dist_' + args.model_name + '.txt', doc_topic_dist, delimiter=' ', fmt='%.4f')
    np.savetxt('../data/' + args.dataset_name + '/results/topic_word_dist_' + args.model_name + '.txt', topic_word_dist, delimiter=' ', fmt='%.4f')


def main(args):

    if args.random_seed:
        np.random.seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)
    train(args)


if __name__ == '__main__':
    main(parse_args())