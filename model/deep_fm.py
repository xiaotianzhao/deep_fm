#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:1/14/19
"""

import tensorflow as tf


class DeepFM(object):
    """DeepFM
    """
    def __init__(self, hps):
        self.hps = hps
        self.cuda_available = tf.test.is_built_with_cuda() and self.hps.cuda
        self.device_name = '/gpu:0' if self.cuda_available else '/cpu:0'

        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=0.1)

    def _add_placeholders(self):
        self.features = tf.placeholder(tf.int32, [None, len(self.hps.field_sizes)])
        self.labels = tf.placeholder(tf.float32, [None, 1])

    def _make_feeddict(self, features, labels):
        feed_dict = {}

        feed_dict[self.features] = features
        feed_dict[self.labels] = labels

        return feed_dict

    def _add_embedding_layer(self):
        with tf.variable_scope("embeddings"):
            self.embedding_matrixs = {}
            for i in range(len(self.hps.field_sizes)):
                self.embedding_matrixs['f2v_{}'.format(i)] = tf.get_variable(
                    'f2v_{}'.format(i),
                    [self.hps.field_sizes[i], self.hps.embedding_size],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )
            self.embeddings_bias_matrixs = {}
            for i in range(len(self.hps.field_sizes)):
                self.embeddings_bias_matrixs['f2bv_{}'.format(i)] = tf.get_variable(
                    'f2bv_{}'.format(i),
                    [self.hps.field_sizes[i], 1],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )

            self.embeddings = []
            for i in range(len(self.hps.field_sizes)):
                self.embeddings.append(
                    tf.nn.embedding_lookup(
                        self.embedding_matrixs['f2v_{}'.format(i)],
                        self.features[:, i]
                    )
                )
            self.embeddings_bias = []
            for i in range(len(self.hps.field_sizes)):
                self.embeddings_bias.append(
                    tf.nn.embedding_lookup(
                        self.embeddings_bias_matrixs['f2bv_{}'.format(i)],
                        self.features[:, i]
                    )
                )
            print(self.embeddings_bias)

    def _add_fm_module(self):
        with tf.variable_scope('fm_component'):
            self.inner_product_results = []
            for i in range(len(self.hps.field_sizes)):
                for j in range(i + 1, len(self.hps.field_sizes)):
                    self.inner_product_results.append(self.embeddings[i] * self.embeddings[j])

            # \sum_{i=1}^{n} w_i * x_j
            print(self.embeddings_bias)
            self.fm_first_order = tf.concat(self.embeddings_bias, axis=-1)
            self.fm_first_order = tf.reduce_sum(self.fm_first_order, axis=-1, keep_dims=True)
            print("FM first Order:", self.fm_first_order)

            # self.fm_cross [batch_size, cross_size, hidden_size]
            self.fm_second_order = tf.concat(
                [tf.expand_dims(inner_product, axis=1) for inner_product in self.inner_product_results],
                axis=1
            )

            print(self.fm_second_order)
            self.fm_second_order = tf.reduce_sum(self.fm_second_order, axis=-1)

            self.fm_outputs = tf.concat((self.fm_first_order, self.fm_second_order), axis=-1)
            print(self.fm_outputs)

    def _add_deep_module(self):
        with tf.variable_scope('deep_component'):
            self.deep_inputs = tf.concat(self.embeddings, axis=-1)
            print(self.hps.field_sizes)
            self.w_1 = tf.get_variable(
                "w1",
                [len(self.hps.field_sizes) * self.hps.embedding_size, self.hps.deep_h1_size],
                initializer=self.trunc_norm_init,
                dtype=tf.float32
            )
            self.b_1 = tf.get_variable(
                "b1",
                [self.hps.deep_h1_size],
                initializer=self.trunc_norm_init,
                dtype=tf.float32
            )

            self.deep_h1 = tf.nn.relu(tf.matmul(self.deep_inputs, self.w_1) + self.b_1)

            self.w_2 = tf.get_variable(
                "w2",
                [self.hps.deep_h1_size, self.hps.deep_h2_size],
                initializer=self.trunc_norm_init,
                dtype=tf.float32
            )
            self.b_2 = tf.get_variable(
                "b2",
                [self.hps.deep_h2_size],
                initializer=self.trunc_norm_init,
                dtype=tf.float32
            )

            self.deep_h2 = tf.nn.relu(tf.matmul(self.deep_h1, self.w_2) + self.b_2)

            self.deep_w_output = tf.get_variable(
                "deep_w_output",
                [self.hps.deep_h2_size, self.hps.deep_h3_size],
                initializer=self.trunc_norm_init,
                dtype=tf.float32
            )

            self.deep_outputs = tf.matmul(self.deep_h2, self.deep_w_output)

    def _add_predict_layer(self):
        with tf.variable_scope("predict_layer"):
            self.outputs = tf.concat((self.fm_outputs, self.deep_outputs), axis=-1)
            print(self.outputs)
            self.pred_w = tf.get_variable(
                "pred_w",
                [self.outputs.get_shape()[-1], 1],
                initializer=self.trunc_norm_init,
                dtype=tf.float32
            )

            self.scores = tf.nn.sigmoid(tf.matmul(self.outputs, self.pred_w))
            print(self.scores)

        with tf.variable_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.scores)
            tf.summary.scalar("loss", self.loss)

    def _add_train_op(self):
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.hps.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        tf.logging.info('Building graph...')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.device(self.device_name):
            self._add_placeholders()
            self._add_embedding_layer()
            self._add_fm_module()
            self._add_deep_module()
            self._add_predict_layer()
            self._add_train_op()

        self.summaries = tf.summary.merge_all()

    def run_train_step(self, sess, features, labels):
        feed_dict = self._make_feeddict(features, labels)
        to_return = {
            "train_op": self.train_op,
            "summaries": self.summaries,
            "loss": self.loss,
            'global_step': self.global_step,
            'predictions': self.scores,
            'labels': self.labels
        }
        return sess.run(to_return, feed_dict)

    def run_dev_step(self, sess, features, labels):
        feed_dict = self._make_feeddict(features, labels)
        to_return = {
            "summaries": self.summaries,
            "loss": self.loss,
            'global_step': self.global_step,
            'predictions': self.scores,
            'labels': self.labels
        }
        return sess.run(to_return, feed_dict)

    def run_test_step(self, sess, features, labels):
        feed_dict = self._make_feeddict(features, labels)
        to_return = {
            "summaries": self.summaries,
            "loss": self.loss,
            'global_step': self.global_step,
            'predictions': self.scores,
            'labels': self.labels
        }
        return sess.run(to_return, feed_dict)
