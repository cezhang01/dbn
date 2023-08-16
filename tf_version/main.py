import argparse
import numpy as np
from model import Model
import os


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, default='dbn', help='dbn or d2bn')
    parser.add_argument('-ne', '--num_training_epoch', type=int, default=100)  # small datasets == 100, large datasets == 5 or 10
    parser.add_argument('-no', '--num_ot_iter', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=64)
    parser.add_argument('-dn', '--dataset_name', type=str, default='ml')
    parser.add_argument('-nt', '--num_topics', type=int, default=64)
    parser.add_argument('-reg_ot', '--reg_ot', type=float, default=2)
    parser.add_argument('-reg_s', '--reg_structure', type=float, default=1)
    parser.add_argument('-reg_kld', '--reg_kld', type=float, default=0.001)
    parser.add_argument('-dp', '--dir_prior', type=float, default=1)
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8)
    parser.add_argument('-wm', '--word_embedding_model', type=str, default='glove')
    parser.add_argument('-wd', '--word_embedding_dimension', type=int, default=300)
    parser.add_argument('-sm', '--structure_embedding_model', type=str, default='deepwalk')
    parser.add_argument('-sd', '--structure_embedding_dimension', type=int, default=128)
    parser.add_argument('-nl', '--num_convolutional_layers', type=int, default=3)
    parser.add_argument('-nn', '--num_sampled_neighbors', type=int, default=5)
    parser.add_argument('-dl', '--percentage_document_length', type=float, default=1)
    parser.add_argument('-rs', '--random_seed', type=int, default=519)
    parser.add_argument('-gpu', '--gpu', type=int, default=0)

    return parser.parse_args()


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.random_seed:
        np.random.seed(args.random_seed)
    model = Model(args)
    print('Start training...')
    model.train()


if __name__ == '__main__':
    main(parse_arguments())