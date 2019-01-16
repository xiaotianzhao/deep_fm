#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:1/14/19
"""

import numpy as np
from deep_fm.utils.vocab import Vocab


class MoviesLenDataset(object):
    """
    Class that iterates over slu Dataset
    __iter__ method yields a tuple(words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is applied

    Example:
        data = MsraDataset(file)
        for sentence,tags in data:
            pass
    """

    def __init__(self, data_type, filename, movieid_vocab, genre_vocab, shuffle=True, random_seed=666):
        """
            Args:
                filename: path to the file
                word_vocab: word vocabulary
                tag_vocab: tag vocabulary
                shuffle: if True, shuffle the data ,else don't preform shuffling
                random_seed: when shuffle the data, which random number you want to use
            """
        self.data_type = data_type
        self.filename = filename
        self.movieid_vocab = movieid_vocab
        self.genre_vocab = genre_vocab
        self.shuffle = shuffle
        self.length = None

    def __iter__(self):
        with open(self.filename, encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            items = line.strip().split('\t')
            userId, movidId, rating, _, genre = items

            yield userId, self.movieid_vocab.word2id(movidId), self.genre_vocab.word2id(genre), float(rating)

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


if __name__ == '__main__':
    genre_vocab = Vocab('../data/genre_vocab', 100000000, False, False, False, False)
    movieid_vocab = Vocab('../data/movie_id_vocab', 100000000, False, False, False, False)

    data = MoviesLenDataset(
        'all',
        '/media/xtzhao/F452BAB352BA7A46/datasets/movies_len/sorted_impl_ratings.csv',
        movieid_vocab,
        genre_vocab
    )

    for index, (userid, movieid, genre, rating) in enumerate(data):
        if index < 100:
            print(userid)
            print(np.asarray(userid).shape)
            features = np.concatenate((np.asarray(userid), np.asarray(movieid), np.asarray(genre)))
            print(features)
        else:
            break