#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:1/10/19
"""

import tensorflow as tf

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
SENTENCE_START = '<SOS>'
SENTENCE_END = '<EOS>'


class Vocab(object):
    def __init__(
        self,
        vocab_file,
        max_size=10000000,
        use_pad=True,
        use_unk=True,
        use_sos=True,
        use_eos=True
    ):
        self.word2idx = {}
        self.count = 0

        if use_pad:
            self.word2idx[PAD_TOKEN] = self.count
            self.count += 1

        if use_unk:
            self.word2idx[UNKNOWN_TOKEN] = self.count
            self.count += 1

        if use_sos:
            self.word2idx[SENTENCE_START] = self.count
            self.count += 1

        if use_eos:
            self.word2idx[SENTENCE_END] = self.count
            self.count += 1

        with open(vocab_file, 'r', encoding='utf8') as vocab_f:
            for index, line in enumerate(vocab_f):
                items = line.strip().split('\t')
                if len(items) != 2:
                    tf.logging.warn('Warning: wrong fromat in file: %s, line: %s' % (vocab_file, line))
                    pass
                else:
                    word, word_cnt = items[0], items[1]
                    self.word2idx[word] = self.count
                    self.count += 1

                if max_size != 0 and self.count >= max_size:
                    tf.logging.info('max_size of vocab was specified as %i; we now have %i words.Stoping reading.' %
                            (max_size, self.count))
        self.idx2word = dict(list(zip(self.word2idx.values(), self.word2idx.keys())))
        tf.logging.info('Finished constructing vocabulary of %i total words. Last word added: %s' %(self.count,
            self.id2word(self.count - 1)))

    def word2id(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNKNOWN_TOKEN]
        return self.word2idx[word]

    def id2word(self, word_id):
        if word_id not in self.idx2word:
            raise ValueError('Id not found in vocab : %d' % word_id)
        return self.idx2word[word_id]

    def size(self):
        return self.count