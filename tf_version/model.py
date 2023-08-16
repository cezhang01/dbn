import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
import collections
from data_loader import Data
from evaluation import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model():

    def __init__(self, args):

        self.data = Data(args)
        self.parse_args(args)
        self.show_config()
        self.generate_placeholders()
        self.generate_variables()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.training_ratio = args.training_ratio
        self.num_docs = self.data.num_docs
        self.num_training_docs = self.data.num_training_docs
        self.num_test_docs = self.data.num_test_docs
        self.percentage_document_length = args.percentage_document_length
        self.num_links = self.data.num_links
        self.num_words = self.data.num_words
        if self.data.labels_available:
            self.num_labels = self.data.num_labels
        self.word_embedding_model = args.word_embedding_model
        self.word_embedding_dimension = args.word_embedding_dimension
        self.structure_embedding_model = args.structure_embedding_model
        self.structure_embedding_dimension = args.structure_embedding_dimension
        self.num_convolutional_layers = args.num_convolutional_layers
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_training_epoch = args.num_training_epoch
        self.num_ot_iter = args.num_ot_iter
        self.learning_rate = args.learning_rate
        self.minibatch_size = args.minibatch_size
        self.num_topics = args.num_topics
        self.gamma = 20
        self.reg_ot = args.reg_ot
        self.reg_structure = args.reg_structure
        self.reg_kld = args.reg_kld
        self.model_name = args.model_name
        self.dir_prior = args.dir_prior
        self.dropout_keep_prob = 0.6
        self.warm_up_epoch = self.num_training_epoch

    def show_config(self):

        print('******************************************************')
        print('tf version:', tf.__version__)
        print('model name:', self.model_name)
        print('dataset name:', self.dataset_name)
        print('training ratio:', self.training_ratio)
        print('#documents:', self.num_docs)
        print('#training documents:', self.num_training_docs)
        print('percentage of document length:', self.percentage_document_length)
        print('#total links:', self.num_links)
        print('#words:', self.num_words)
        if self.data.labels_available:
            print('#labels:', self.num_labels)
        print('word embedding model:', self.word_embedding_model)
        print('dimension of word embeddings:', self.word_embedding_dimension)
        print('structure embedding model:', self.structure_embedding_model)
        print('dimension of structure embeddings:', self.structure_embedding_dimension)
        print('#convolutional layers:', self.num_convolutional_layers)
        print('#sampled neighbors:', self.num_sampled_neighbors)
        print('#training epochs:', self.num_training_epoch)
        print('#ot iterations:', self.num_ot_iter)
        print('learning rate:', self.learning_rate)
        print('minibatch size:', self.minibatch_size)
        print('#topics:', self.num_topics)
        print('regularizer for ot:', self.reg_ot)
        print('regularizer for structure:', self.reg_structure)
        print('regularizer for kl divergence:', self.reg_kld)
        print('Dirichlet prior:', self.dir_prior)
        print('******************************************************')

    def generate_placeholders(self):

        self.placeholders = collections.defaultdict(list)
        self.placeholders['doc_ids'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])
        self.placeholders['neighbor_docs_bow'] = tf.placeholder(dtype=tf.float32, shape=[self.minibatch_size * self.num_sampled_neighbors, self.num_words])
        if self.model_name == 'd2bn':
            self.placeholders['neighbor_docs_adj'] = tf.placeholder(dtype=tf.float32, shape=[self.minibatch_size * self.num_sampled_neighbors, self.num_training_docs])
        self.placeholders['dropout_keep_prob'] = tf.placeholder(tf.float32)
        self.placeholders['warm_up'] = tf.placeholder(tf.float32)

    def generate_variables(self):

        self.variables = collections.defaultdict(list)
        self.convolution_dim = [self.word_embedding_dimension] + [128] * (self.num_convolutional_layers - 1) + [self.num_topics]

    def evaluate_topic_word_cost_matrix(self):

        word_embeddings_norm = tf.nn.l2_normalize(self.word_embeddings, axis=1)
        topic_embeddings_norm = tf.nn.l2_normalize(self.variables['topic_embeddings'], axis=1)
        self.topic_word_cost_matrix = 1 - tf.matmul(topic_embeddings_norm, tf.transpose(word_embeddings_norm))  # [num_topics, num_words]

    def evaluate_topic_structure_cost_matrix(self):

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.variables['adj_w'] = tf.get_variable(
                name='adj_w',
                shape=[self.structure_embedding_dimension, self.word_embedding_dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        structure_embeddings = tf.matmul(self.structure_embeddings, tf.squeeze(self.variables['adj_w']))
        structure_embeddings_norm = tf.nn.l2_normalize(structure_embeddings, axis=1)
        topic_embeddings_norm = tf.nn.l2_normalize(tf.stop_gradient(self.variables['topic_embeddings']), axis=1)
        self.topic_structure_cost_matrix = 1 - tf.matmul(tf.stop_gradient(topic_embeddings_norm), tf.transpose(structure_embeddings_norm))

    def sinkhorn_algorithm(self, dist_1, dist_2, cost_matrix):

        psi_1 = tf.ones_like(dist_1) / tf.cast(tf.shape(dist_1)[-1], tf.float32)
        psi_2 = tf.ones_like(dist_2) / tf.cast(tf.shape(dist_2)[-1], tf.float32)
        h = tf.exp(- cost_matrix * self.gamma)
        for _ in range(self.num_ot_iter):
            psi_2 = tf.multiply(dist_2, 1 / (tf.matmul(psi_1, h) + 1e-12))
            psi_1 = tf.multiply(dist_1, 1 / (tf.matmul(psi_2, tf.transpose(h)) + 1e-12))
        ot_distances = tf.reduce_sum(tf.multiply(tf.matmul(psi_1, tf.multiply(h, cost_matrix)), psi_2), axis=-1)

        self.mean_ot_loss = tf.reduce_sum(ot_distances) / tf.cast(tf.shape(dist_1)[0], tf.float32)

        return ot_distances

    def message_passing(self, self_embeds, neighbor_embeds, layer_id, act):

        with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
            self.variables['conv_' + str(layer_id)] = tf.get_variable(
                name='conv_' + str(layer_id),
                shape=[self.convolution_dim[layer_id], self.convolution_dim[layer_id + 1]],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        self_embeds = tf.nn.dropout(self_embeds, keep_prob=self.placeholders['dropout_keep_prob'])
        self_embeds = tf.matmul(self_embeds, self.variables['conv_' + str(layer_id)])
        neighbor_embeds = tf.nn.dropout(neighbor_embeds, keep_prob=self.placeholders['dropout_keep_prob'])
        neighbor_embeds = tf.matmul(neighbor_embeds, self.variables['conv_' + str(layer_id)])
        self.attention_values = self.evaluate_attention(self_embeds, neighbor_embeds, layer_id)
        neighbor_embeds_agg, self_embeds_agg = self.neighbor_aggregation(self_embeds, neighbor_embeds, self.attention_values, act)

        return self_embeds_agg

    def evaluate_attention(self, self_embeds, neighbor_embeds, layer_id):

        with tf.variable_scope('att', reuse=tf.AUTO_REUSE):
            self.variables['att_' + str(layer_id)] = tf.get_variable(
                name='att_' + str(layer_id),
                shape=[2 * self.convolution_dim[layer_id + 1], 1],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        num_neighbors = tf.cast(tf.divide(tf.shape(neighbor_embeds)[0], tf.shape(self_embeds)[0]), tf.int32)
        embeds_repeat = tf.reshape(tf.tile(tf.expand_dims(self_embeds, axis=1), [1, num_neighbors, 1]), tf.shape(neighbor_embeds))
        attention_values = tf.matmul(tf.concat([embeds_repeat, neighbor_embeds], axis=1), self.variables['att_' + str(layer_id)])
        attention_values = tf.reshape(attention_values, [-1, num_neighbors])
        attention_values = tf.nn.softmax(tf.nn.sigmoid(attention_values))

        return attention_values

    def neighbor_aggregation(self, self_embeds, neighbor_embeds, attention_values, act):

        attention_values = tf.expand_dims(attention_values, axis=1)
        neighbor_embeds = tf.reshape(neighbor_embeds, [tf.shape(attention_values)[0], tf.shape(attention_values)[-1], tf.shape(neighbor_embeds)[-1]])
        neighbor_embeds_agg = tf.squeeze(tf.matmul(attention_values, neighbor_embeds))
        eta = 0.5
        self_embeds_agg = act(eta * self_embeds + (1 - eta) * neighbor_embeds_agg)
        neighbor_embeds_agg = act(neighbor_embeds_agg)

        return neighbor_embeds_agg, self_embeds_agg

    def get_neighbors(self, doc_ids):

        neighbors = collections.defaultdict(list)
        neighbors['docs'] = [doc_ids]
        for layer_id in range(self.num_convolutional_layers):
            neighbor_doc_ids = tf.reshape(tf.gather(self.data.sampled_neighbors, neighbors['docs'][layer_id]), [-1])
            neighbors['docs'].append(neighbor_doc_ids)

        return neighbors

    def reparameterization(self, alpha):

        self.B = 10
        gamma = tf.squeeze(tf.random_gamma(shape=[1], alpha=alpha + tf.to_float(self.B)))
        epsilon = tf.stop_gradient(tf.sqrt(9. * (alpha + tf.to_float(self.B)) - 3.) * (tf.pow(gamma / (alpha + tf.to_float(self.B) - 1. / 3.), 1. / 3.) - 1.))
        u = tf.random_uniform([self.B, self.minibatch_size, self.num_topics])
        gamma = self.gamma_rejection_sampling_boosted(alpha, epsilon, u)
        doc_topic_dist = tf.div(gamma, tf.reduce_sum(gamma, axis=1, keepdims=True))
        doc_topic_dist.set_shape(alpha.get_shape())

        return doc_topic_dist

    def gamma_rejection_sampling(self, alpha, epsilon):

        b = alpha - 1. / 3.
        c = 1. / tf.sqrt(9. * b)
        v = 1. + epsilon * c

        return b * (v ** 3)

    def gamma_rejection_sampling_boosted(self, alpha, epsilon, u):

        i = tf.to_float(tf.reshape(tf.range(self.B), [-1, 1, 1]))  # [B, 1, 1]
        alpha_vec = tf.reshape(tf.tile(alpha, [self.B, 1]), [self.B, -1, self.num_topics]) + i  # [B, minibatch_size, num_topics] + [B, 1, 1]
        u_pow = tf.pow(u, 1. / alpha_vec) + 1e-10
        gamma_tmp = self.gamma_rejection_sampling(alpha + tf.to_float(self.B), epsilon)
        gamma = tf.reduce_prod(u_pow, axis=0) * gamma_tmp

        return gamma

    def graph_convolution(self, doc_ids):

        self.neighbors = self.get_neighbors(doc_ids)

        doc_embeds = []
        for layer_id in range(self.num_convolutional_layers + 1):
            doc_embeds.append(tf.nn.embedding_lookup(self.docs, self.neighbors['docs'][layer_id]))

        for layer_id in range(self.num_convolutional_layers):
            if layer_id == self.num_convolutional_layers - 1:
                act = lambda x: x
            else:
                act = tf.nn.tanh
            next_doc_embeds = []
            for hop in range(self.num_convolutional_layers - layer_id):
                doc_embeds_agg = self.message_passing(self_embeds=doc_embeds[hop], neighbor_embeds=doc_embeds[hop + 1], layer_id=layer_id, act=act)
                next_doc_embeds.append(doc_embeds_agg)
            doc_embeds = next_doc_embeds

        mean = doc_embeds[0]
        mean = tf.contrib.layers.batch_norm(mean)
        alpha = tf.maximum(1e-12, tf.log(1. + tf.exp(mean)))
        doc_topic_dist = self.reparameterization(alpha)
        prior = tf.ones([self.minibatch_size, self.num_topics], dtype=tf.float32) * self.dir_prior
        #kld = tf.reduce_mean(tf.contrib.distributions.Dirichlet(alpha).log_prob(doc_topic_dist + 1e-12) - tf.contrib.distributions.Dirichlet(prior).log_prob(doc_topic_dist + 1e-12))

        kld = tf.lgamma(tf.reduce_sum(alpha, axis=1)) - tf.lgamma(tf.reduce_sum(prior, axis=1))
        kld -= tf.reduce_sum(tf.lgamma(alpha), axis=1)
        kld += tf.reduce_sum(tf.lgamma(prior), axis=1)
        minus = alpha - prior
        test = tf.reduce_sum(tf.multiply(minus, tf.digamma(alpha) - tf.reshape(tf.digamma(tf.reduce_sum(alpha, 1)), [self.minibatch_size, 1])), axis=1)
        kld += test
        kld = tf.reduce_mean(kld)

        return doc_topic_dist, kld, mean

    def optimal_transport(self, doc_topic_dist, y_true, mode='topic_word'):

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.variables['topic_embeddings'] = tf.get_variable(
                name='topic_embeddings',
                shape=[self.num_topics, self.word_embedding_dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        doc_topic_dist_repeat = tf.reshape(tf.tile(doc_topic_dist, [1, self.num_sampled_neighbors]), [-1, self.num_topics])
        y_true_norm = tf.nn.softmax(y_true)
        if mode == 'topic_word':
            self.evaluate_topic_word_cost_matrix()
            ot_distance = self.sinkhorn_algorithm(doc_topic_dist_repeat, y_true_norm, self.topic_word_cost_matrix)
        else:
            self.evaluate_topic_structure_cost_matrix()
            ot_distance = self.sinkhorn_algorithm(doc_topic_dist_repeat, y_true_norm, self.topic_structure_cost_matrix)
        ot_distance = tf.reshape(ot_distance, [-1, self.num_sampled_neighbors])
        ot_distance = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.attention_values, ot_distance), axis=-1))

        return ot_distance

    def decoder(self, doc_topic_dist, y_true, decoder_para):

        self.y_pred = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.matmul(doc_topic_dist, decoder_para)))
        y_pred = tf.reshape(tf.tile(tf.expand_dims(self.y_pred, axis=1), [1, self.num_sampled_neighbors, 1]), tf.shape(y_true))
        y_true = y_true / tf.tile(tf.reduce_sum(y_true, axis=-1, keepdims=True), [1, tf.shape(y_true)[-1]])
        recon_loss = - tf.reduce_sum(tf.multiply(y_true, tf.log(y_pred + 1e-12)), axis=-1)  # maybe change to cross entropy in tf library
        recon_loss = tf.reshape(recon_loss, [-1, self.num_sampled_neighbors])
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.attention_values, recon_loss), axis=-1))

        return recon_loss

    def l2_loss(self):

        l2_loss = 0
        for layer_id in range(self.num_convolutional_layers):
            l2_loss += tf.nn.l2_loss(self.variables['conv_' + str(layer_id)])

        return l2_loss

    def construct_model(self):

        self.docs = tf.Variable(tf.constant(self.data.docs, dtype=tf.float32), trainable=False)
        self.word_embeddings = tf.Variable(tf.constant(self.data.word_embeddings, dtype=tf.float32), trainable=False)
        if self.model_name == 'd2bn':
            self.structure_embeddings = tf.Variable(tf.constant(self.data.structure_embeddings, dtype=tf.float32), trainable=False)

        loss = 0
        doc_topic_dist, kld, self.mean = self.graph_convolution(self.placeholders['doc_ids'])
        if self.model_name == 'd2bn':
            loss += self.reg_structure * self.reg_ot * self.optimal_transport(doc_topic_dist, self.placeholders['neighbor_docs_adj'], 'topic_structure')
            loss += self.reg_structure * self.decoder(doc_topic_dist, self.placeholders['neighbor_docs_adj'], 2 - self.topic_structure_cost_matrix)
        loss += self.reg_ot * self.optimal_transport(doc_topic_dist, self.placeholders['neighbor_docs_bow'], 'topic_word')
        loss += self.decoder(doc_topic_dist, self.placeholders['neighbor_docs_bow'], 2 - self.topic_word_cost_matrix)
        loss += self.reg_kld * self.placeholders['warm_up'] * kld
        loss += 0 * self.l2_loss()

        return loss

    def prepare_feed_dict(self, minibatch_idx, total_doc_ids):

        feed_dict = {}
        doc_ids = total_doc_ids[minibatch_idx * self.minibatch_size:(minibatch_idx + 1) * self.minibatch_size]
        if len(doc_ids) < self.minibatch_size:
            indices = np.random.choice(len(total_doc_ids), self.minibatch_size - len(doc_ids), replace=True)
            doc_ids = np.concatenate([doc_ids, total_doc_ids[indices]])

        feed_dict[self.placeholders['doc_ids']] = doc_ids
        neighbor_doc_ids = np.reshape(self.data.sampled_neighbors[doc_ids], [-1])
        feed_dict[self.placeholders['neighbor_docs_bow']] = self.data.generate_bow(neighbor_doc_ids)
        if self.model_name == 'd2bn':
            feed_dict[self.placeholders['neighbor_docs_adj']] = self.data.generate_adj(neighbor_doc_ids)
        feed_dict[self.placeholders['dropout_keep_prob']] = self.dropout_keep_prob
        feed_dict[self.placeholders['warm_up']] = self.warm_up

        return feed_dict

    def train(self):

        loss = self.construct_model()
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)
            num_minibatch = int(np.ceil(self.num_training_docs / self.minibatch_size))
            t = time.time()
            one_epoch_loss = 0
            for epoch_idx in range(1, self.num_training_epoch + 1):
                self.data.sample_neighbors()
                self.warm_up = np.minimum(1, float(epoch_idx) / self.warm_up_epoch)
                for minibatch_idx in range(num_minibatch):
                    feed_dict = self.prepare_feed_dict(minibatch_idx, np.arange(self.num_training_docs))
                    _, one_epoch_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
                if epoch_idx % 5 == 0 or epoch_idx == 1:
                    print('******************************************************')
                    print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_idx, self.num_training_epoch), '\tLoss: %f' % one_epoch_loss)

                    # infer training doc_topic_dist
                    doc_topic_dist_training = []
                    num_minibatch_eval = int(np.ceil(self.num_training_docs / self.minibatch_size))
                    for minibatch_idx in range(num_minibatch_eval):
                        feed_dict = self.prepare_feed_dict(minibatch_idx, np.arange(self.num_training_docs))
                        feed_dict.update({self.placeholders['dropout_keep_prob']: 1})
                        doc_topic_dist_training.extend(sess.run(self.mean, feed_dict=feed_dict))

                    # infer test doc embeds
                    doc_topic_dist_test, doc_word_dist_test = [], []
                    num_minibatch_eval = int(np.ceil(self.num_test_docs / self.minibatch_size))
                    for minibatch_idx in range(num_minibatch_eval):
                        feed_dict = self.prepare_feed_dict(minibatch_idx, np.arange(self.num_training_docs, self.num_docs))
                        feed_dict.update({self.placeholders['dropout_keep_prob']: 1})
                        doc_topic_dist_test_tmp, doc_word_dist_test_tmp = sess.run([self.mean, self.y_pred], feed_dict=feed_dict)
                        doc_topic_dist_test.extend(doc_topic_dist_test_tmp)
                        doc_word_dist_test.extend(doc_word_dist_test_tmp)
                    doc_topic_dist_training = np.array(doc_topic_dist_training)[:self.num_training_docs]
                    doc_topic_dist_test = np.array(doc_topic_dist_test)[:self.num_test_docs]
                    doc_topic_dist = np.concatenate([doc_topic_dist_training, doc_topic_dist_test], axis=0)
                    doc_word_dist_test = np.array(doc_word_dist_test)[:self.num_test_docs]

                    # evaluation
                    if self.data.labels_available:
                        classification_knn(doc_topic_dist_training, doc_topic_dist_test, self.data.training_labels, self.data.test_labels)
                        clustering_kmeans(doc_topic_dist_training, doc_topic_dist_test, self.data.training_labels, self.data.test_labels)
                    link_prediction_auc(doc_topic_dist_test, self.data.test_links, self.num_training_docs)

                    y_true = self.data.generate_bow(np.arange(self.num_training_docs, self.num_docs), normalize=False)
                    perplexity(doc_word_dist_test, y_true)

                    topic_word_dist = 2 - sess.run(self.topic_word_cost_matrix)
                    topic_diversity(topic_word_dist)

            output_top_words(topic_word_dist, self.data.voc)
            # save model outputs
            folder = os.path.exists('../data/' + self.dataset_name + '/results')
            if not folder:
                os.makedirs('../data/' + self.dataset_name + '/results')
            np.savetxt('../data/' + self.dataset_name + '/results/doc_topic_dist_' + self.model_name + '.txt', doc_topic_dist, delimiter=' ', fmt='%.4f')
            np.savetxt('../data/' + self.dataset_name + '/results/topic_word_dist_' + self.model_name + '.txt', topic_word_dist, delimiter=' ', fmt='%.4f')