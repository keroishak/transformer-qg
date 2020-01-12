import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import json
import re
import os
import argparse

def normalize_text(text):
    text = text.replace('-RRB-','')
    text = text.replace('-LRB-','')
    text = re.sub('[()\"#/@;:<>{}`+=~|!]_', '', text)
    return text.lower().strip()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(topic, paragraph, answer, question):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'topic': _bytes_feature(topic.encode()),
      'paragraph': _bytes_feature(paragraph.encode()),
      'answer': _bytes_feature(answer.encode()),
      'question': _bytes_feature(question.encode())
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()    
def write_tf_records(examples, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for example in examples:
            writer.write(example)

def convert_to_tf_records(input_file):
    examples_list=[]
    with open(input_file) as json_file:
        data = json.load(json_file)
        topic_list = data['data']
        unlabeled_answer_segment=0
        for topic in topic_list:
            topic_title = topic['title']
            topic_title_text=normalize_text(topic_title)
            for paragraph in topic['paragraphs']:
                paragraph_text = normalize_text(paragraph['context'])
                for question in paragraph['qas']:
                    question_text = normalize_text(question['question'])
                    answer = question['answers'][0]
                    answer_text = normalize_text(answer['text'])
                    examples_list.append(serialize_example(topic_title_text, paragraph_text, answer_text, question_text))
    dataset_len=len(examples_list)
    examples_numpy_arr = np.array(examples_list)
    np.random.shuffle(examples_numpy_arr)
    training_examples = examples_numpy_arr[0:int(0.8*dataset_len)]
    validation_examples = examples_numpy_arr[int(0.8*dataset_len)::]
    splitted_input_file = input_file.split('/')
    validation_output_file = '/'.join(splitted_input_file[0:len(splitted_input_file)-1]) + '/dev_dataset.tfrecord'
    training_output_file = '/'.join(splitted_input_file[0:len(splitted_input_file)-1]) + '/train_dataset.tfrecord'

    write_tf_records(validation_examples, validation_output_file)
    write_tf_records(training_examples, training_output_file)
    
    special_tokens=['<PAD>', '<SEP>', '<SOS>', '<EOS>', '<ANS>']
    vocab_text=[]
    for row in training_examples:
            example = tf.train.Example()
            example.ParseFromString(row)
            vocab_text.append(example.features.feature['paragraph'].bytes_list.value[0].decode('utf-8'))
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        corpus_generator =(txt for txt in vocab_text), reserved_tokens=special_tokens, target_vocab_size=2**13) 
    tokenizer.save_to_file('./wordpieces')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='path to data file', required=True)
    args = parser.parse_args()
    input_file = args.file_path
    assert os.path.exists(input_file) is not None, "%s file is not found" % (input_file)
    convert_to_tf_records(input_file)
if __name__=='__main__':
    main()