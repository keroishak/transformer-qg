import tensorflow as tf
from custom_schedule import CustomSchedule
from transformer import Transformer
from input import Input
from config import Config
import argparse
import datetime

def get_loss_function():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def calculate_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = get_loss_function()
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
        
    return tf.reduce_mean(loss_)  

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len) 

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask       

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
class TrainingManager:
  
  def __init__(self, config):
    self.config = config
    self.input = Input(config)
    learning_rate = CustomSchedule(self.config.d_model, warmup_steps=self.config.warmup_steps)
    # self.optimizer = tf.keras.optimizers.SGD(learning_rate)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                      epsilon=1e-9)
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          name='train_accuracy')

    self.dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    self.dev_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          name='dev_accuracy')      

    self.transformer = Transformer(self.config.num_layers, self.config.d_model, self.config.num_heads, self.config.dff,
                            vocab_size=self.input.tokenizer.vocab_size, 
                            pe_input=self.input.tokenizer.vocab_size, 
                            pe_target=self.config.target_sentence_max_len,
                            rate=self.config.dropout_rate)    

    ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                optimizer=self.optimizer)

    self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.config.checkpoint_dir, max_to_keep=5)

      # if a checkpoint exists, restore the latest checkpoint.
    if self.ckpt_manager.latest_checkpoint and self.config.restore == True:
      ckpt.restore(self.ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = self.config.summary_dir + '/' + current_time + '/train'
    dev_log_dir = self.config.summary_dir + '/' + current_time + '/dev'
    self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    self.dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)  

  @tf.function(input_signature=train_step_signature)
  def train_step(self, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
      predictions, _ = self.transformer(inp, tar_inp, 
                                  True, 
                                  enc_padding_mask, 
                                  combined_mask, 
                                  dec_padding_mask,
                                  self.config.temperature)
      loss = calculate_loss(tar_real, predictions)

    gradients = tape.gradient(loss, self.transformer.trainable_variables)    
    self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
    
    self.train_loss(loss)
    self.train_accuracy(tar_real, predictions)
  
  @tf.function(input_signature=train_step_signature)
  def dev_step(self, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    predictions, _ = self.transformer(inp, tar_inp, 
                                False, 
                                enc_padding_mask, 
                                combined_mask, 
                                dec_padding_mask,
                                self.config.temperature)
    loss = calculate_loss(tar_real, predictions)
    
    self.dev_loss(loss)
    self.dev_accuracy(tar_real, predictions)

  def train(self):
    for epoch in range(self.config.epoch):
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      batch=0
      for inp, target in self.input.train_ds:
        self.train_step(inp, target)
        batch = batch + 1
        if batch % 100 == 0:
          print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4%}'.format(
            epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))  

      with self.train_summary_writer.as_default():
        tf.summary.histogram("Train Loss", self.train_loss.result(), step=epoch+1)
        tf.summary.histogram("Train Accuracy", self.train_accuracy.result(), step=epoch+1)
        # tf.summary.histogram("LR", self.learning_rate.lr, step=epoch)

      self.dev_loss.reset_states()
      self.dev_accuracy.reset_states()
      for inp, target in self.input.dev_ds:
        self.dev_step(inp, target)

      with self.dev_summary_writer.as_default():
        tf.summary.histogram("Dev Loss", self.dev_loss.result(), step=epoch+1)
        tf.summary.histogram("Dev Accuracy", self.dev_accuracy.result(), step=epoch+1)

      print ('Epoch {} Dev Loss {:.4f} Dev Accuracy {:.4%}'.format(
                    epoch + 1, self.dev_loss.result(), self.dev_accuracy.result()))
      if (epoch + 1) % 5 == 0:
          ckpt_save_path = self.ckpt_manager.save()
          print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                              ckpt_save_path))
       

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--vocab_path', type=str, help='path to subwords file', required=True)
  parser.add_argument('--train_file', type=str, help='path to the training TF-record dataset file', required=True)
  parser.add_argument('--dev_file', type=str, help='path to the development TF-record dataset file', required=True)
  parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
  parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
  parser.add_argument('--d_model', type=int, default=512, help='model dimension')
  parser.add_argument('--num_layers', type=int, default=6, help='number of stacked layers')
  parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
  parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup steps')
  parser.add_argument('--temperature', type=float, default=0.6, help='softmax temperature value')
  parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='dir path to save model(ex, ./checkpoint)')
  parser.add_argument('--restore', type=bool, default=False, help='boolean flag whether to restore the latest checkpoint or not')
  parser.add_argument('--summary_dir', type=str, default='./runs', help='path to save summary(ex, ./runs)')
  args = parser.parse_args()

  config = Config(args)
  trainingManager = TrainingManager(config)
  trainingManager.train()

        
        