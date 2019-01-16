#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:1/14/19
"""

import os
import json
import numpy as np
import tensorflow as tf
from collections import namedtuple
# from sklearn.metrics import roc_auc_score

from deep_fm.utils.vocab import Vocab
from deep_fm.utils.metrics import auc_score
from deep_fm.model.deep_fm import DeepFM
from deep_fm.utils.dataset import MoviesLenDataset
from deep_fm.utils.batcher import BatcherMoviesLen

# Where to find data
tf.app.flags.DEFINE_string('data_path', '/media/xtzhao/F452BAB352BA7A46/datasets/movies_len/sorted_impl_ratings.csv', '')
tf.app.flags.DEFINE_string('word_vocab_path', './data/snips_joint/word_vocab', '')
tf.app.flags.DEFINE_string('tag_vocab_path', './data/snips_joint/tag_vocab', '')
tf.app.flags.DEFINE_string('intent_tag_vocab_path', './data/snips_joint/intent_tag_vocab', '')
tf.app.flags.DEFINE_string('w2v_matrix_path', './data/snips_joint/w2v_matrix', '')

tf.app.flags.DEFINE_string('output', './data/temp', '')
tf.app.flags.DEFINE_string('script', './eval/conlleval.pl', '')
tf.app.flags.DEFINE_string('config', '', 'config file path')

# Important settings
tf.app.flags.DEFINE_string('model', 'deep_fm', 'must be one of deep_fm/')
tf.app.flags.DEFINE_string('field_sizes', '138493, 27278, 20', 'field sizes')
tf.app.flags.DEFINE_boolean('cuda', True, 'if cuda is True and gpu is available, use gpu.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', './data/log', 'Root directory for all logging')
tf.app.flags.DEFINE_string('exp_name', 'deep_fm', 'Name for experiment')

# Training settings
tf.app.flags.DEFINE_integer('print_every', 100, 'number of epoch')
tf.app.flags.DEFINE_integer('save_every', 15, 'number of save_every')
tf.app.flags.DEFINE_integer('early_stop_num', 100, 'early stop number')

# Hyperparameters
tf.app.flags.DEFINE_integer('epoch_num', 1, 'number of epoch')
tf.app.flags.DEFINE_integer('embedding_size', 24, 'dimension of embedding')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer("deep_h1_size", 100, "")
tf.app.flags.DEFINE_integer("deep_h2_size", 100, "")
tf.app.flags.DEFINE_integer("deep_h3_size", 100, "")
tf.app.flags.DEFINE_integer("lucky_num", 666, "")
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg_lambda', 0, 'l2 regularize lambda')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_float('keep_dropout_prob', 0.5, 'Keep dropout prob')


FLAGS = tf.app.flags.FLAGS


def get_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def avg_auc(average, cnt, new_value):
    return ((cnt - 1) / cnt) * average + new_value / cnt


def calc_avg_loss(average_loss, loss, decay=0.99):
    if average_loss is None:
        average_loss = loss
    else:
        average_loss = average_loss * decay + (1 - decay) * loss
    return average_loss


def write2summary(value, tag_name, train_step, summary_writer):
    summary = tf.Summary()
    summary.value.add(tag=tag_name, simple_value=value)
    summary_writer.add_summary(summary, train_step)


def run_train_and_test(model, data, genre_vocab, movieid_vocab, hps):
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=get_config())
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(hps.log_root)
    bestmodel_save_path = os.path.join(hps.log_root, "bestmodel")
    avg_loss = None
    avg_auc_score = 0
    cnt = 0
    result_fop = open(os.path.join(hps.log_root, 'result'), 'w', encoding='utf8')

    config_fop = open(os.path.join(hps.log_root, 'config.json'), 'w', encoding='utf8')
    json.dump(hps._asdict(), config_fop)

    try:
        for i in range(hps.epoch_num):
            batcher = BatcherMoviesLen()
            # tf.logging.info("slot loss weight:{:.4f}".format(slot_loss_weight))

            for index, batch in enumerate(batcher.batch(data, hps.batch_size)):
                user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, true_batch_size = batch
                features = np.concatenate(
                    (
                        np.asarray(user_id_batch).reshape(-1, 1),
                        np.asarray(movie_id_batch).reshape(-1, 1),
                        np.asarray(movie_genre_batch).reshape(-1, 1)
                    ),
                    axis=1
                )
                rating_batch = np.asarray(rating_batch).reshape([-1, 1])
                results = model.run_train_step(sess, features, rating_batch)

                loss, train_step, summaries = results['loss'], results['global_step'], results['summaries']
                avg_loss = calc_avg_loss(avg_loss, loss)

                if train_step > 2000:
                    auc_cur = auc_score(
                        list(zip(results['predictions'].reshape(-1).tolist(), rating_batch.reshape(-1).tolist()))
                    )
                    avg_auc_score = avg_auc(avg_auc_score, cnt + 1, auc_cur)
                    cnt += 1
                    write2summary(auc_cur, 'metrics/cur_auc', train_step, summary_writer)

                write2summary(avg_loss, 'running_avg_loss/decay=0.99', train_step, summary_writer)
                write2summary(loss, 'metrics/cur_loss', train_step, summary_writer)
                write2summary(avg_auc_score, 'metrics/avg_auc', train_step, summary_writer)

                # tf.logging.info(data_utils.calc_metrics(results['predictions'], tag_ids, tag_vocab, masks))
                if train_step % hps.print_every == 0:
                    tf.logging.info("Epoch {}".format(i))
                    tf.logging.info("Batch {}".format(index + 1))
                    tf.logging.info("Loss {}".format(avg_loss))
                    if train_step > 2000:
                        tf.logging.info("AVE-AUC: {}, CUR-AUC: {}".format(avg_auc_score, auc_cur))
                    tf.logging.info("==========================================")
                    summary_writer.flush()
    except KeyboardInterrupt:

        exit(0)


def main(argv):
    """Program Start Method"""
    tf.logging.set_verbosity(tf.logging.INFO)

    hps = {}
    if tf.__version__ == '1.12.0':
        tf.logging.info(type(FLAGS.__dict__['__wrapped']))
        for key, value in FLAGS.__flags.items():
            hps[key] = value.value
    else:
        for key, value in FLAGS.__dict__['__flags'].items():
            hps[key] = value

    if os.path.exists(FLAGS.config):
        configs = json.load(open(FLAGS.config, 'r', encoding='utf8'))
        for key, value in configs.items():
            hps[key] = value
    tf.set_random_seed(hps['lucky_num'])

    tf.logging.info("Loading vocabulary...")
    genre_vocab = Vocab('./data/genre_vocab', 100000000, False, False, False, False)
    movieid_vocab = Vocab('./data/movie_id_vocab', 100000000, False, False, False, False)

    tf.logging.info("Vocabulary loaded!")

    tf.logging.info("Loading data...")
    data = MoviesLenDataset(
        'all',
        hps['data_path'],
        movieid_vocab,
        genre_vocab
    )
    tf.logging.info("Data loaded!")
    hps['log_root'] = os.path.join(hps['log_root'], hps['exp_name'])

    for key, value in hps.items():
        tf.logging.info("Parameters: %s - %s" % (key, value))

    hps['field_sizes'] = [int(field_size) for field_size in hps['field_sizes'].split(',')]
    hps = namedtuple('HParams', hps.keys())(**hps)

    if hps.model == 'deep_fm':
        model = DeepFM(hps)
    else:
        raise NotImplementedError

    run_train_and_test(model, data, genre_vocab, movieid_vocab, hps)


if __name__ == '__main__':
    tf.app.run()