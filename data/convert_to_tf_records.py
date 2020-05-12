import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import json
import re
import os
import argparse
import stanfordnlp
import spacy
from tokenization import FullTokenizer

nlp = spacy.load("en_core_web_sm")

# os.environ["CORENLP_HOME"] = "../../stanford-corenlp/"
# stanfordnlp.download('en')
# stanfordnlp.Pipeline()


def normalize_text(text):
    text = text.replace('-RRB-', '')
    text = text.replace('-LRB-', '')
    text = re.sub("[\\[#|/@;:<>{}`+=~\!\\]_]", '<>', text)
    index = text.find('<>')
    # the following postprocessing to handle cases like "here i'm ....."
    while index > -1:
        if index > 0 and text[index - 1] == ' ' or index + 2 < len(
                text) and text[index + 2] == ' ':
            text = text.replace('<>', '', 1)
        else:
            text = text.replace('<>', ' ', 1)
        index = text.find('<>')
    return text.strip()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns a int64_list from a int."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(paragraph, answer, question, answer_start_idx):
    """
  Creates a tf.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'paragraph': _bytes_feature(paragraph.encode()),
        'answer': _bytes_feature(answer.encode()),
        'question': _bytes_feature(question.encode()),
        'answer_start_idx': _int64_feature(answer_start_idx)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))
    return example_proto.SerializeToString()


def write_tf_records(examples, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for example in examples:
            writer.write(example)


def trim_paragraph(paragraph_text, answer_text):
    paragraph_sentences = nltk.sent_tokenize(paragraph_text)
    answer_sentences = nltk.sent_tokenize(answer_text)

    for i in range(len(paragraph_sentences)):
        sentence = paragraph_sentences[i]
        if len(answer_sentences) > 1 and answer_sentences[0] in sentence:
            #Asumption: if the tokenizer splitted the answer into more than two sentences, it means that the same applied on paragraph text on the same text segment
            answer_found = True
            for j in range(1, len(answer_sentences)):
                if answer_sentences[j] not in paragraph_sentences[i + j]:
                    answer_found = False
                    break
            if answer_found:
                answer_sentence_offset = paragraph_text.find(
                    paragraph_sentences[i + len(answer_sentences) - 1])
                return paragraph_text[:answer_sentence_offset + len(
                    paragraph_sentences[i + len(answer_sentences) - 1])]

        else:
            if answer_text in sentence:
                answer_sentence_offset = paragraph_text.find(sentence)
                if answer_sentence_offset == -1:
                    print(
                        'Error: something went wrong. can\'t find answer sentence in the paragraph text'
                    )
                    exit(1)

                return paragraph_text[:answer_sentence_offset + len(sentence)]
    print('Error: something went wrong. can\'t find answer in the paragraph')
    exit(1)


def get_answer_start_token(paragraph_tokens, answer_tokens):
    if len(answer_tokens) == 0:
        print(paragraph_tokens)
        print(answer_tokens)
        return -1
    for i in range(len(paragraph_tokens)):
        if paragraph_tokens[i] == answer_tokens[0]:
            temp_iterator = i + 1
            for j in range(1, len(answer_tokens)):
                if paragraph_tokens[temp_iterator] != answer_tokens[j]:
                    break
                temp_iterator += 1
            if (temp_iterator - i) == len(answer_tokens):
                return i
    return -1


def get_tokens(text, tokenizer):
    return tokenizer.tokenize(text)


def number_mapping(text):
    if text >= '0' and text <= '9':
        if text == '0':
            return 'zero'
        elif text == '1':
            return 'one'
        elif text == '2':
            return 'two'
        elif text == '3':
            return 'three'
        elif text == '4':
            return 'four'
        elif text == '5':
            return 'five'
        elif text == '6':
            return 'six'
        elif text == '7':
            return 'seven'
        elif text == '8':
            return 'eight'
        elif text == '9':
            return 'nine'

    return "NaN"


def convert_to_tf_records(input_file, vocab_file_path, trim_up_to_answer,
                          replace_answer_span):
    examples_list = []
    tokenizer = FullTokenizer(vocab_file=vocab_file_path)
    malformed_data = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        topic_list = data['data']
        unlabeled_answer_segment = 0
        for topic in topic_list:
            topic_title = topic['title']
            topic_title_text = normalize_text(topic_title)
            for paragraph in topic['paragraphs']:
                paragraph_text = normalize_text(paragraph['context'])
                paragraph_tokens = get_tokens(paragraph_text, tokenizer)
                encoded_paragraph_text = ' '.join(paragraph_tokens)
                for question in paragraph['qas']:
                    question_text = normalize_text(question['question'])
                    answer = question['answers'][0]
                    answer_text = normalize_text(answer['text'])
                    answer_start_ids = answer['answer_start']

                    end_idx = paragraph['context'].find(
                        ' ', answer_start_ids + len(answer_text) + 5)
                    answer_sentence_snippet = paragraph['context'][
                        answer_start_ids:end_idx]
                    answer_sentence_snippet = normalize_text(
                        answer_sentence_snippet)
                    answer_sentence_snippet_tokens = get_tokens(
                        answer_sentence_snippet, tokenizer)

                    answer_token_start = get_answer_start_token(
                        paragraph_tokens, answer_sentence_snippet_tokens)
                    if answer_token_start == -1:
                        malformed_data.append({
                            'paragraph':
                            paragraph['context'],
                            'questions':
                            question,
                            'answer_text':
                            answer['text'],
                            'answer_start':
                            answer['answer_start']
                        })
                        # print("something went wrong")
                        # print(paragraph_tokens)
                        # print(answer_sentence_snippet_tokens)
                        # print(answer['text'])
                        # print(answer['answer_start'])
                        continue
                    answer_tokens = get_tokens(answer_text, tokenizer)
                    question_tokens = get_tokens(question_text, tokenizer)

                    encoded_answer_text = ' '.join(answer_tokens)
                    encoded_question_text = ' '.join(question_tokens)
                    examples_list.append(
                        serialize_example(encoded_paragraph_text,
                                          encoded_answer_text,
                                          encoded_question_text,
                                          answer_token_start))
    print(len(malformed_data))
    with open('malformed_data.json', 'w') as malformed_data_json:
        json.dump(malformed_data, malformed_data_json)
    dataset_len = len(examples_list)
    examples_numpy_arr = np.array(examples_list)
    np.random.shuffle(examples_numpy_arr)
    train_examples = examples_numpy_arr[0:int(0.8 * dataset_len)]
    dev_examples = examples_numpy_arr[int(0.8 * dataset_len):int(0.9 *
                                                                 dataset_len)]
    test_examples = examples_numpy_arr[int(0.9 * dataset_len)::]
    splitted_input_file = input_file.split('/')

    train_output_file = '/'.join(
        splitted_input_file[0:len(splitted_input_file) -
                            1]) + '/train_dataset.tfrecord'
    dev_output_file = '/'.join(
        splitted_input_file[0:len(splitted_input_file) -
                            1]) + '/dev_dataset.tfrecord'
    test_output_file = '/'.join(
        splitted_input_file[0:len(splitted_input_file) -
                            1]) + '/test_dataset.tfrecord'

    write_tf_records(train_examples, train_output_file)
    write_tf_records(dev_examples, dev_output_file)
    write_tf_records(test_examples, test_output_file)

    # special_tokens = ['<PAD>', '<SEP>', '<SOS>', '<EOS>', '<ANS>']
    # vocab_text = []
    # for row in training_examples:
    #     example = tf.train.Example()
    #     example.ParseFromString(row)
    #     vocab_text.append(
    #         example.features.feature['paragraph'].bytes_list.value[0].decode(
    #             'utf-8'))
    # tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #     corpus_generator=(txt for txt in vocab_text),
    #     reserved_tokens=special_tokens,
    #     target_vocab_size=2**13)
    # tokenizer.save_to_file('./wordpieces')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path',
                        type=str,
                        help='path to the json data file',
                        required=True)
    parser.add_argument('--vocab_file_path',
                        type=str,
                        help='path to the voacb file',
                        required=True)
    parser.add_argument(
        '--trim_up_to_answer',
        type=str2bool,
        help='boolean flag whether to trim up to answer sentence',
        default=False,
        required=True)
    parser.add_argument(
        '--replace_answer_span',
        type=str2bool,
        default=False,
        help='boolean flag whether to replace answer span by <ANS> tag',
        required=True)
    args = parser.parse_args()
    input_file = args.json_file_path
    vocab_file_path = args.vocab_file_path
    trim_up_to_answer = args.trim_up_to_answer
    replace_answer_span = args.replace_answer_span
    assert os.path.exists(input_file) is not None, "%s file is not found" % (
        input_file)
    assert os.path.exists(
        vocab_file_path) is not None, "%s file is not found" % (
            vocab_file_path)
    convert_to_tf_records(input_file, vocab_file_path, trim_up_to_answer,
                          replace_answer_span)


if __name__ == '__main__':
    main()