import tensorflow as tf
import tensorflow_datasets as tfds

class Input():
    def __init__(self, config):
        self.EOS = 3
        self.train_ds = tf.data.TFRecordDataset([config.train_ds_file])
        self.dev_ds = tf.data.TFRecordDataset([config.dev_ds_file])
        self.train_ds = self.train_ds.map(self.__tf_encode)
        self.dev_ds = self.dev_ds.map(self.__tf_encode)
        self.train_ds = self.train_ds.cache().padded_batch(config.batch_size, padded_shapes=([-1], [-1]))
        self.train_ds = self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.dev_ds = self.dev_ds.padded_batch(config.batch_size, padded_shapes=([-1], [-1]))

        
        self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.subwords_file)
    
    def encode(self, txt_record):
        example = tf.train.Example()
        example.ParseFromString(txt_record.numpy())
        paragraph = example.features.feature['paragraph'].bytes_list.value[0].decode('utf-8')
        answer = example.features.feature['answer'].bytes_list.value[0].decode('utf-8')
        question = example.features.feature['question'].bytes_list.value[0].decode('utf-8')
        input_txt = '<SOS>' + answer + '<SEP>' + paragraph + '<EOS>'
        question = '<SOS>' + question + '<EOS>'
        return self.tokenizer.encode(input_txt), self.tokenizer.encode(question)

    def __tf_encode(self, txt_record):
        return tf.py_function(self.encode, [txt_record], [tf.int64, tf.int64])

if __name__=='__main__':
    input = Input('./dataset/training_dataset.tfrecord', './dataset/validation_dataset.tfrecord')

    