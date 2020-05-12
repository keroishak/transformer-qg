import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from embedding import Embedding


class Transformer(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        num_layers = config['num_hidden_layers']
        d_model = config['hidden_size']
        num_heads = config['num_attention_heads']
        dff = config['intermediate_size']
        vocab_size = config['vocab_size']
        max_tokens_len = config['max_position_embeddings']
        rate = config['hidden_dropout_prob']

        embedding = Embedding(config)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, embedding,
                               rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, embedding,
                               rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, answer_mask, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(
            inp, answer_mask, training,
            enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training,
                                                     look_ahead_mask,
                                                     dec_padding_mask)

        final_output = self.final_layer(
            dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
