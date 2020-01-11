import tensorflow as tf
import json
import re
import os
import argparse

def normalize_text(text):
    text = text.replace('-RRB-','')
    text = text.replace('-LRB-','')
    text = re.sub('[()\"#/@;:<>{}`+=~|!]_', '', text)
    return text
def flatten_data_into_raws(input_file):
    splitted_input_file = input_file.split('.')
    output_file = '.'.join(splitted_input_file[0:len(splitted_input_file)-1]) + '.txt'
    data_file = open(output_file, 'w')
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
                    data_file.write(topic_title_text + '|' + paragraph_text + '|' + answer_text + '|' + question_text + '\n' )
        data_file.close()
        description_file_path = '.'.join(splitted_input_file[0:len(splitted_input_file)-1]) + '_description.txt'
        description_file = open(description_file_path, 'w')
        description_file.write('topic_title | paragraph_text | answer_text | question_text')
        description_file.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='path to data file', required=True)
    args = parser.parse_args()
    input_file = args.file_path
    assert os.path.exists(input_file) is not None, "%s file is not found" % (input_file)
    flatten_data_into_raws(input_file)
if __name__=='__main__':
    main()