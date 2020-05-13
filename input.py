import tensorflow as tf
import tensorflow_datasets as tfds
from data.tokenization import FullTokenizer
import numpy as np


class Input():
    def __init__(self, config):
        self.max_len = config['max_position_embeddings']
        self.train_ds = tf.data.TFRecordDataset([config['train_ds_file']])
        self.dev_ds = tf.data.TFRecordDataset([config['dev_ds_file']])
        self.test_ds = tf.data.TFRecordDataset([config['test_ds_file']])
        self.train_ds = self.train_ds.map(self.__tf_encode).filter(
            self.filter_max_length)
        self.dev_ds = self.dev_ds.map(self.__tf_encode).filter(
            self.filter_max_length)
        self.train_ds = self.train_ds.cache().padded_batch(
            config['batch_size'], padded_shapes=([-1], [-1], [-1]))
        self.train_ds = self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.dev_ds = self.dev_ds.padded_batch(config['batch_size'],
                                               padded_shapes=([-1], [-1], [-1
                                                                           ]))

        self.tokenizer = FullTokenizer(vocab_file=config['bert_dir'] +
                                       '/vocab.txt')

    def filter_max_length(self, inp, answer_mask, question):
        return tf.less_equal(tf.size(inp), self.max_len)

    def encode(self, txt_record):
        example = tf.train.Example()
        example.ParseFromString(txt_record.numpy())
        paragraph = example.features.feature['paragraph'].bytes_list.value[
            0].decode('utf-8').split(' ')
        answer = example.features.feature['answer'].bytes_list.value[0].decode(
            'utf-8').split(' ')
        question = example.features.feature['question'].bytes_list.value[
            0].decode('utf-8').split(' ')
        answer_start_idx_str = example.features.feature[
            'answer_start_idx'].bytes_list.value[0].decode('utf-8')
        answer_start_idx =  int(answer_start_idx_str)
        input_txt = ['[CLS]'] + paragraph + ['[SEP]']
        answer_len = len(answer)
        question = ['[CLS]'] + question + ['[SEP]']
        answer_mask = np.zeros(len(input_txt), dtype=int)
        answer_mask[answer_start_idx:answer_start_idx + answer_len] = 1
        return self.tokenizer.convert_tokens_to_ids(
            input_txt), answer_mask, self.tokenizer.convert_tokens_to_ids(
                question)

    def __tf_encode(self, txt_record):
        return tf.py_function(self.encode, [txt_record],
                              [tf.int64, tf.int64, tf.int64])
