#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:1/13/19
"""

import random
import numpy as np
from deep_fm.utils.vocab import Vocab
from deep_fm.utils.dataset import MoviesLenDataset

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
SENTENCE_START = '<SOS>'
SENTENCE_END = '<EOS>'


class BatcherMoviesLen(object):
    def __init__(self, batcher_name="movieslen_batcher", random_seed=666):
        self.batcher_name = batcher_name
        self.random_seed = random_seed

    def minibatches(self, data, minibatch_size):
        user_id_batch, movie_id_batch, movie_genre_batch, rating_batch = [], [], [], []
        for user_id, movie_id, movie_genre, rating in data:
            if len(user_id_batch) == minibatch_size:
                yield user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, minibatch_size
                user_id_batch, movie_id_batch, movie_genre_batch, rating_batch = [], [], [], []

            user_id_batch += [user_id]
            movie_id_batch += [movie_id]
            movie_genre_batch += [movie_genre]
            rating_batch += [rating]

        if len(user_id_batch) == minibatch_size:
            yield user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, minibatch_size
        else:
            true_batch_size = len(user_id_batch)
            while len(user_id_batch) != minibatch_size:
                rand_idx = random.randint(0, len(user_id_batch) - 1)
                user_id_batch += [user_id_batch[rand_idx]]
                movie_id_batch += [movie_id_batch[rand_idx]]
                movie_genre_batch += [movie_genre_batch[rand_idx]]
                rating_batch += [rating_batch[rand_idx]]

            yield user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, true_batch_size

    def batch(self, data, minibatch_size):
        for batch in self.minibatches(data, minibatch_size):
            user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, true_batch_size = batch
            yield user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, true_batch_size


class BatcherClassification(object):
    """
    分类任务的Batch生成器
    """
    def __init__(self, batcher_name="batcher_classification", random_seed=666):
        self.batcher_name = batcher_name
        random.seed(random_seed)

    def minibatches(self, data, minibatch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
        Returns:
            list of tuples
        """
        x_batch, y_batch = [], []
        for (x, y) in data:
            if len(x_batch) == minibatch_size:
                indexs = list(range(len(x_batch)))
                indexs.sort(key=lambda l: -len(x_batch[l]))
                x_batch = [x_batch[l] for l in indexs]
                y_batch = [y_batch[l] for l in indexs]

                yield x_batch, y_batch, minibatch_size
                x_batch, y_batch = [], []
            x_batch += [x]
            y_batch += [y]

        if len(x_batch) == minibatch_size:
            indexs = list(range(len(x_batch)))
            indexs.sort(key=lambda l : -len(x_batch[l]))
            x_batch = [x_batch[l] for l in indexs]
            y_batch = [y_batch[l] for l in indexs]

            yield x_batch, y_batch, minibatch_size
        else:
            true_batch_size = len(x_batch)
            while len(x_batch) != minibatch_size:
                rand_idx = random.randint(0, len(x_batch) - 1)
                x_batch.append(x_batch[rand_idx])
                y_batch.append(y_batch[rand_idx])

            indexs.sort(key=lambda l: -len(x_batch[l]))
            x_batch = [x_batch[l] for l in indexs]
            y_batch = [y_batch[l] for l in indexs]

            yield x_batch, y_batch, true_batch_size

    def _pad_sequences(self, sequences, pad_tok, max_length):
        """
            Args:
                sequences: a generator of list or tuple
                pad_tok: the char to pad with
            Returns:
                a list of list where each sublist has same length
            """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    def pad_sequences(self, sequences, pad_tok):
        """
            Args:
                sequences: a generator of list or tuple
                pad_tok: the char to pad with
            Returns:
                a list of list where each sublist has same length
            """
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = self._pad_sequences(sequences, pad_tok, max_length)
        return sequence_padded, sequence_length


class BatcherClsSeq(object):
    """
    序列，分类联合任务的Batch生成器
    """
    def __init__(self, batcher_name="batcher_classification", random_seed=666):
        self.batcher_name = batcher_name
        random.seed(random_seed)

    def minibatches(self, data, minibatch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
        Returns:
            list of tuples
        """
        orig_x_batch, x_batch, y_batch, z_batch = [], [], [], []
        for (orig_x, x, y, z) in data:
            if len(x_batch) == minibatch_size:
                indexs = list(range(len(x_batch)))
                indexs.sort(key=lambda l: -len(x_batch[l]))

                orig_x_batch = [orig_x_batch[l] for l in indexs]
                x_batch = [x_batch[l] for l in indexs]
                y_batch = [y_batch[l] for l in indexs]
                z_batch = [z_batch[l] for l in indexs]

                yield orig_x_batch, x_batch, y_batch, z_batch, minibatch_size
                orig_x_batch, x_batch, y_batch, z_batch = [], [], [], []
            orig_x_batch += [orig_x]
            x_batch += [x]
            y_batch += [y]
            z_batch += [z]

        if len(x_batch) == minibatch_size:
            indexs = list(range(len(x_batch)))
            indexs.sort(key=lambda l: -len(x_batch[l]))

            orig_x_batch = [orig_x_batch[l] for l in indexs]
            x_batch = [x_batch[l] for l in indexs]
            y_batch = [y_batch[l] for l in indexs]
            z_batch = [z_batch[l] for l in indexs]

            yield orig_x_batch, x_batch, y_batch, z_batch, minibatch_size
        else:
            true_batch_size = len(x_batch)
            while len(x_batch) != minibatch_size:
                rand_idx = random.randint(0, len(x_batch) - 1)

                orig_x_batch.append(orig_x_batch[rand_idx])
                x_batch.append(x_batch[rand_idx])
                y_batch.append(y_batch[rand_idx])
                z_batch.append(z_batch[rand_idx])

            indexs = list(range(len(x_batch)))
            indexs.sort(key=lambda l: -len(x_batch[l]))

            orig_x_batch = [orig_x_batch[l] for l in indexs]
            x_batch = [x_batch[l] for l in indexs]
            y_batch = [y_batch[l] for l in indexs]
            z_batch = [z_batch[l] for l in indexs]

            yield orig_x_batch, x_batch, y_batch, z_batch, true_batch_size

    def _pad_sequences(self, sequences, pad_tok, max_length):
        """
            Args:
                sequences: a generator of list or tuple
                pad_tok: the char to pad with
            Returns:
                a list of list where each sublist has same length
            """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    def pad_sequences(self, sequences, pad_tok):
        """
            Args:
                sequences: a generator of list or tuple
                pad_tok: the char to pad with
            Returns:
                a list of list where each sublist has same length
            """
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = self._pad_sequences(sequences, pad_tok, max_length)
        return sequence_padded, sequence_length

    def batch(self, data, minibatch_size, word_pad_token, tag_pad_token, max_seq_len):
        for minibatch in self.minibatches(data, minibatch_size=minibatch_size):
            ori_words, words, tags, labels, true_batch_size = minibatch
            if max_seq_len == -1:
                words, seq_lens = self.pad_sequences(words, word_pad_token)
                tags, tag_seq_lens = self.pad_sequences(tags, tag_pad_token)
            else:
                words, seq_lens = self._pad_sequences(words, word_pad_token, max_seq_len)
                tags, tag_seq_lens = self._pad_sequences(tags, tag_pad_token, max_seq_len)

            assert (seq_lens == tag_seq_lens)

            yield ori_words, words, tags, labels, seq_lens, true_batch_size


if __name__ == '__main__':

    genre_vocab = Vocab('../data/genre_vocab', 100000000, False, False, False, False)
    movieid_vocab = Vocab('../data/movie_id_vocab', 100000000, False, False, False, False)

    data = MoviesLenDataset(
        'all',
        '/media/xtzhao/F452BAB352BA7A46/datasets/movies_len/sorted_impl_ratings.csv',
        movieid_vocab,
        genre_vocab
    )

    batcher = BatcherMoviesLen()
    for batch in batcher.batch(data, 64):
        user_id_batch, movie_id_batch, movie_genre_batch, rating_batch, true_batch_size = batch

        features = np.concatenate(
            (
                np.asarray(user_id_batch).reshape(-1, 1),
                np.asarray(movie_id_batch).reshape(-1, 1),
                np.asarray(movie_genre_batch).reshape(-1, 1)
            ),
            axis=1
        )
        print(features.shape)

