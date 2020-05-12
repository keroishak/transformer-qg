import tensorflow as tf
from modules import create_initializer, positional_encoding


class Embedding(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        vocab_size = config['vocab_size']
        self.d_model = config['hidden_size']
        answer_token_type_size = config['answer_token_type_size']
        self.maximum_position_encoding = config['max_position_embeddings']
        hidden_dropout_prob = config['hidden_dropout_prob']

        self.word_embeddings = tf.keras.layers.Embedding(
            vocab_size,
            self.d_model,
            embeddings_initializer=create_initializer(
                config['initializer_range']),
            name='word_embeddings')
        self.pos_encoding = positional_encoding(self.maximum_position_encoding,
                                                self.d_model)
        self.answer_token_type_embeddings = tf.keras.layers.Embedding(
            answer_token_type_size,
            self.d_model,
            embeddings_initializer=create_initializer(
                config['initializer_range']),
            name='answer_token_type_embeddings')

        self.layernorm = tf.keras.layers.LayerNormalization(name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_prob)

    def call(self, input, answer_token_type_ids=None, training=False):
        seq_len = tf.shape(input)[1]
        # adding embedding and position encoding.
        final_embeddings = self.word_embeddings(
            input)  # (batch_size, input_seq_len, d_model)

        max_len_assert_op = tf.debugging.assert_less_equal(
            seq_len,
            self.maximum_position_encoding,
            message='input exceeded max len %d' %
            (self.maximum_position_encoding))
        with tf.control_dependencies([max_len_assert_op]):
            # adding position encoding.
            final_embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            final_embeddings += self.pos_encoding[:, :seq_len, :]

        if answer_token_type_ids is not None:
            final_embeddings += self.answer_token_type_embeddings(
                answer_token_type_ids)

        final_embeddings = self.layernorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings, training=training)

        return final_embeddings

    def get_config(self):
        return {
            'word_embeddings': self.word_embeddings,
            'answer_token_type_embeddings': self.answer_token_type_embeddings
        }
