import argparse
import copy
import numpy as np
import os
import random
import json
import tensorflow as tf
from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate

tf.compat.v1.disable_eager_execution()
from time import time

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops


    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


class ReRaPrLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name, parameters, steps=1, epochs=50, batch_size=None, flat=False, gpu=True, in_pro=False,
                 new_relation_graph=None):

        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        self.new_relation_graph = new_relation_graph

        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        # relation data
        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}
        self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(self.data_path, '..', 'relation', self.relation_name,
                         self.market_name + rname_tail[self.relation_name])
        )
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)

        #self.embedding = np.load(
        #    os.path.join(self.data_path, '..', 'pretrain', emb_fname))
        #print('embedding shape:', self.embedding.shape)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu

        # Process new_relation_graph
        self.new_relation_graph = new_relation_graph
        self.new_rel_mask = self.construct_new_relation_graph(new_relation_graph)
        #print('relation mask shape:',self.rel_mask.shape)
        #print('new relation mask shape:', self.new_rel_mask.shape)
        #print(self.rel_mask)
        #print(self.new_rel_mask)

    def construct_new_relation_graph(self, json_file):
        with open(json_file, 'r') as f:
            new_graph_data = json.load(f)

        # Assuming the JSON file contains a dictionary where each key is a stock symbol
        # and the value is a list of related stock symbols
        relation_dict = new_graph_data

        # Get the tickers (stock symbols) from the relation dictionary
        new_tickers = list(relation_dict.keys())

        # Make sure that new_tickers only contains the tickers that are in the old graph
        common_tickers = [ticker for ticker in new_tickers if ticker in self.tickers]

        # Initialize an empty relation matrix
        num_tickers = len(common_tickers)
        relation_encoding = np.ones((num_tickers, num_tickers), dtype=np.float32) * -1e9

        # Construct the relation encoding matrix based on the relation dictionary
        for i, ticker in enumerate(common_tickers):
            related_tickers = relation_dict[ticker]
            for related_ticker in related_tickers:
                if related_ticker in common_tickers:
                    j = common_tickers.index(related_ticker)
                    relation_encoding[i, j] = 0

        return relation_encoding

    def combine_masks(self, relation_mask, new_relation_mask):
        boolean_relation_mask = tf.cast(relation_mask != -1e9, dtype=tf.bool)
        boolean_new_relation_mask = tf.cast(new_relation_mask != -1e9, dtype=tf.bool)

        combined_mask = tf.where(
            boolean_relation_mask & (relation_mask == new_relation_mask),
            1.0,
            tf.where(
                boolean_relation_mask & (relation_mask != new_relation_mask),
                0.0,
                tf.where(
                    boolean_new_relation_mask,
                    -1e9,
                    relation_mask
                )
            )
        )

        return combined_mask

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    def train(self):
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.compat.v1.reset_default_graph()

            seed = 123456789
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            ground_truth = tf.compat.v1.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.compat.v1.placeholder(tf.float32, [self.batch_size, 1])
            feature = tf.compat.v1.placeholder(tf.float32,[self.batch_size, self.parameters['seq'], self.fea_dim])
            base_price = tf.compat.v1.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)

            lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
                self.parameters['unit']
            )

            initial_state = lstm_cell.zero_state(self.batch_size,
                                                 dtype=tf.float32)
            outputs, _ = tf.compat.v1.nn.dynamic_rnn(
                lstm_cell, feature, dtype=tf.float32,
                initial_state=initial_state
            )
            seq_emb = outputs[:, -1, :]

            relation = tf.constant(self.rel_encoding, dtype=tf.float32)
            rel_mask = tf.constant(self.rel_mask, dtype=tf.float32)

            new_relation_mask = tf.constant(self.new_rel_mask, dtype=tf.float32)


            # Combine the two masks
            #combined_mask = self.combine_masks(rel_mask, new_relation_mask)
            #combined_mask = new_relation_mask

            #rel_weight = tf.compat.v1.layers.dense(relation, units=1,
            #                             activation=leaky_relu)
            rel_weight = tf.keras.layers.Dense(units=1, activation=leaky_relu)(relation)

            if self.inner_prod:
                print('inner product weight')
                inner_weight = tf.matmul(seq_emb, seq_emb, transpose_b=True)
                weight = tf.multiply(inner_weight, rel_weight[:, :, -1])
            else:
                print('sum weight')
                #head_weight = tf.compat.v1.layers.dense(feature, units=1,
                #                              activation=leaky_relu)
                head_weight = tf.keras.layers.Dense(units=1, activation=leaky_relu)(seq_emb)
                #tail_weight = tf.compat.v1.layers.dense(feature, units=1,
                #                              activation=leaky_relu)
                tail_weight = tf.keras.layers.Dense(units=1, activation=leaky_relu)(seq_emb)
                weight = tf.add(
                    tf.add(
                        tf.matmul(head_weight, all_one, transpose_b=True),
                        tf.matmul(all_one, tail_weight, transpose_b=True)
                    ), rel_weight[:, :, -1]
                )

            weight_price = tf.nn.softmax(tf.add(tf.cast(new_relation_mask, tf.float32), weight), axis=0)
            weight_masked = tf.nn.softmax(tf.add(tf.cast(rel_mask, tf.float32), weight), axis=0)

            outputs = tf.matmul(weight_price, seq_emb)
            outputs_proped = tf.matmul(weight_masked, outputs)
            if self.flat:
                print('one more hidden layer')
                #outputs_concated = tf.compat.v1.layers.dense(
                #    tf.concat([feature, outputs_proped], axis=1),
                #    units=self.parameters['unit'], activation=leaky_relu,
                #    kernel_initializer=tf.compat.v1.glorot_uniform_initializer()
                #)
                concatenated = tf.concat([seq_emb, outputs_proped], axis=1)
                outputs_concated = tf.keras.layers.Dense(units=self.parameters['unit'], activation=leaky_relu,
                            kernel_initializer=tf.compat.v1.glorot_uniform_initializer())(concatenated)

            else:
                outputs_concated = tf.concat([seq_emb, outputs_proped], axis=1)

            # One hidden layer
            #prediction = tf.compat.v1.layers.dense(
            #    outputs_concated, units=1, activation=leaky_relu, name='reg_fc',
            #    kernel_initializer=tf.compat.v1.glorot_uniform_initializer()
            #)
            prediction = tf.keras.layers.Dense(units=1, activation=leaky_relu, name='reg_fc',
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer())(outputs_concated)

            return_ratio = tf.divide(tf.subtract(prediction, base_price), base_price)
            reg_loss = tf.compat.v1.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )
            pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio, transpose_b=True)
            )
            gt_pw_dif = tf.subtract(
                tf.matmul(all_one, ground_truth, transpose_b=True),
                tf.matmul(ground_truth, all_one, transpose_b=True)
            )
            mask_pw = tf.matmul(mask, mask, transpose_b=True)
            rank_loss = tf.reduce_mean(
                tf.nn.relu(
                    tf.multiply(
                        tf.multiply(pre_pw_dif, gt_pw_dif),
                        mask_pw
                    )
                )
            )
            loss = reg_loss + tf.cast(self.parameters['alpha'], tf.float32) * \
                              rank_loss
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.parameters['lr']
            ).minimize(loss)
        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_test_pred = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_gt = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_mask = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_valid_perf = {
            'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
        }
        best_test_perf = {
            'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
        }
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.valid_index - self.parameters['seq'] -
                                   self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, batch_out = \
                    sess.run((loss, reg_loss, rank_loss, optimizer),
                             feed_dict)
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss
            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))


            # test on validation set
            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            for cur_offset in range(
                self.valid_index - self.parameters['seq'] - self.steps + 1,
                self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr, = \
                    sess.run((loss, reg_loss, rank_loss,
                              return_ratio), feed_dict)
                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss
                cur_valid_pred[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index -
                                              self.parameters['seq'] -
                                              self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            print('Valid MSE:',
                  val_loss / (self.test_index - self.valid_index),
                  val_reg_loss / (self.test_index - self.valid_index),
                  val_rank_loss / (self.test_index - self.valid_index))
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                      cur_valid_mask)
            print('\t Valid preformance:', cur_valid_perf)

            # test on testing set
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            for cur_offset in range(
                                            self.test_index - self.parameters['seq'] - self.steps + 1,
                                            self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = \
                    sess.run((loss, reg_loss, rank_loss,
                              return_ratio), feed_dict)
                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss

                cur_test_pred[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            print('Test MSE:',
                  test_loss / (self.trade_dates - self.test_index),
                  test_reg_loss / (self.trade_dates - self.test_index),
                  test_rank_loss / (self.trade_dates - self.test_index))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            print('\t Test performance:', cur_test_perf)
            if val_loss / (self.test_index - self.valid_index) < \
                    best_valid_loss:
                best_valid_loss = val_loss / (self.test_index -
                                              self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)
                print('Better valid loss:', best_valid_loss)
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)
        sess.close()
        tf.compat.v1.reset_default_graph()
        return best_valid_pred, best_valid_gt, best_valid_mask, \
               best_test_pred, best_test_gt, best_test_mask, \
               best_valid_perf, best_test_perf

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    #physical_devices = tf.config.list_physical_devices('GPU')
    #for device in physical_devices:
    #    tf.config.experimental.set_memory_growth(device, True)
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='use gpu')

    parser.add_argument('-e', '--emb_file', type=str,
                        default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
                        help='fname for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='wikidata',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    parser.add_argument('-ct', default=0.95,
                        help='correlation_threshold')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    RRP_LSTM = ReRaPrLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        parameters=parameters,
        steps=1, epochs=50, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod,
        new_relation_graph='../data/price_graph/'+args.m+'_correlation_graph_'+str(args.ct)+'.json'
    )

    pred_all = RRP_LSTM.train()
    #print('Pred_All:',pred_all)

#Best Valid performance: {'mse': 0.0004958068088824255, 'mrrt': 0.0199531754167635, 'btl': 2.3591825100884307}
#Best Test performance: {'mse': 0.00037784899098915913, 'mrrt': 0.027345894375467396, 'btl': 0.9153886415879242}