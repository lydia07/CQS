import json
import tqdm
import nltk
from tqdm import tqdm
from collections import defaultdict
from keywords import (calc_tfidf, get_keywords)
from data_utils import (word_tokenize, merge_keywords, get_spans, find_closest_span,
                        clean_str)

total_keywords_num = 'total_keywords_num'
keywords_per_question = 'keywords_per_question'
keywords_per_paragraph = 'keywords_per_paragraph'
total_data_num = 'total_data_num'
total_question_num = 'total_question_num'
keywords_per_article = 'keywords_per_article'
question_per_paragraph = 'question_per_paragraph'
question_per_article = 'question_per_article'
total_paragraphs_num = 'total_paragraphs_num'
total_aricles_num = 'total_aricles_num'
avg_ques_len = 'avg_ques_len'
avg_para_len = 'avg_para_len'
max_ques_len = 'max_ques_len'
max_para_len = 'max_para_len'


class DataStat:

    def __init__(self, data_dir):
        print(data_dir)
        self.data = self.load_data(data_dir)
        self.avg_keywords = 0
        # self.avg_context_len = 0
        self.avg_questions = 0
        self.one_question = 0
        self.two_question = 0
        self.above_three_question = 0
        self.paragraph_num = 0
        self.keywords_num = 0
        self.get_stat()

    def load_data(self, data_dir):
        with open(data_dir, 'r') as f:
            data = json.load(f)
            return data

    def get_stat(self):
        total_length = 0
        total_keywords = 0
        total_questions = 0
        self.paragraph_num = len(self.data)
        for paragraph in self.data:
            # print(paragraph)
            # print('keywords num:{}'.format(len(paragraph['keywords'])))
            # total_length += len(paragraph['context'])
            total_keywords += len(paragraph['keywords'])
            for keyword in paragraph['keywords'].keys():
                # print('keyword:{}'.format(keyword))
                question_num = len(paragraph['keywords'][keyword])
                # print('question num under {}:{}'.format(keyword, question_num))
                total_questions += question_num
                if question_num == 1:
                    self.one_question += 1
                elif question_num == 2:
                    self.two_question += 1
                else:
                    self.above_three_question +=1
        self.avg_context_len = total_length / self.paragraph_num
        self.keywords_num = total_keywords
        self.avg_keywords = total_keywords / self.paragraph_num
        self.avg_questions = total_questions / total_keywords

    def show_stat(self):
        print('-------------------------------------')
        print('Total num of paragraghs:{}'.format(self.paragraph_num))
        # print('Average paragraph length:{}'.format(self.avg_context_len))
        print('Average number of keywords:{}'.format(self.avg_keywords))
        print('Questions per keyword:{}'.format(self.avg_questions))
        print('Keyword with 1 question:{},{:.2f} of total'.format(self.one_question, self.one_question / self.keywords_num))
        print('Keyword with 2 questions:{},{:.2f} of total'.format(self.two_question, self.two_question / self.keywords_num))
        print('Keyword with above 3 questions:{},{:.2f} of total'.format(self.above_three_question, self.above_three_question / self.keywords_num))
        print('-------------------------------------')

def datastat(file_path, ratio=0.5):
    with open(file_path, 'r') as f:
        load_dict = json.load(f)
        data = load_dict['data']
    sents = []
    for title in tqdm(data):
        for para in title['paragraphs']:
            for qa in para['qas']:
                if qa['is_impossible']:
                    continue
                sents.append(clean_str(qa['question']))
    tfidf, vocab = calc_tfidf(sents, word_tokenize)
    data_stat = {
                        'total_keywords_num': 0,
                        'keywords_per_question': 0,
                        'keywords_per_paragraph': 0,
                        'total_data_num': 0,
                        'total_question_num': 0,
                        'keywords_per_article': 0,
                        'question_per_article': 0,
                        'total_paragraphs_num': 0,
                        'total_aricles_num': 0,
                        'question_per_paragraph': 0,
                        'avg_ques_len': 0,
                        'avg_para_len': 0,
                        'max_ques_len': 0,
                        'max_para_len': 0,
                    }
    counter = defaultdict(lambda: 0)
    pks = []
    total = 0
    question_idx = 0
    data_stat[total_aricles_num] = len(data)
    for title in tqdm(data):
        data_stat[total_paragraphs_num] += len(title['paragraphs'])
        for para in title['paragraphs']:
            context = clean_str(para['context'])
            context_tokens = word_tokenize(context)
            for token in context_tokens:
                counter[token] += 1
            # spans = convert_idx(context, context_tokens)
            data_stat[avg_para_len] += len(context_tokens)
            if len(context_tokens) > data_stat[max_para_len]:
                data_stat[max_para_len] = len(context_tokens)
            for qa in para['qas']:
                if qa['is_impossible']:
                    continue
                total += 1
                question = clean_str(qa['question'])
                question_tokens = word_tokenize(question)

                for token in question_tokens:
                    counter[token] += 1
                keywords = get_keywords(question_idx, question, word_tokenize, tfidf, 
                                        vocab, ratio)
                keywords = merge_keywords(question, keywords)
                
                question_idx += 1
                
                
                if not keywords:
                    # data_stat['not_generated'] += 1
                    continue
                data_stat[total_question_num] += 1
                data_stat[avg_ques_len] += len(question_tokens)
                if len(question_tokens) > data_stat[max_ques_len]:
                    data_stat[max_ques_len] = len(question_tokens)

                for keyword in keywords:
                    answer = clean_str(qa['answers'][0]['text'])
                    answer_spans = get_spans(context_tokens, answer)
                    keyword_spans = get_spans(context_tokens, keyword)
                    if not (answer_spans and keyword_spans):
                        # data_stat['not_found'] += 1
                        continue 
                    data_stat[total_keywords_num] += 1
                    answer_span = answer_spans[0]
                    
                    keyword_span = find_closest_span(answer_span, keyword_spans)
                    pk = {
                        'context_tokens': context_tokens,
                        'question_tokens': question_tokens,
                        'start': keyword_span['start'],
                        'end': keyword_span['end'],
                        'keyword': keyword
                    }
                    pks.append(pk)
                
    # data_stat['lost_total'] = data_stat['not_generated'] + data_stat['not_found']
    data_stat[total_data_num] = len(pks)
    data_stat[keywords_per_question] = data_stat[total_keywords_num] / data_stat[total_question_num]
    data_stat[keywords_per_paragraph] = data_stat[total_keywords_num] / data_stat[total_paragraphs_num]
    data_stat[keywords_per_article] = data_stat[total_keywords_num] / data_stat[total_aricles_num]
    data_stat[question_per_article] = data_stat[total_question_num] / data_stat[total_aricles_num]
    data_stat[question_per_paragraph] = data_stat[total_question_num] / data_stat[total_paragraphs_num]
    data_stat[avg_ques_len] = data_stat[avg_ques_len] / data_stat[total_question_num]
    data_stat[avg_para_len] = data_stat[avg_para_len] / data_stat[total_paragraphs_num]
    print(data_stat)

if __name__ == '__main__':
    # datastat = DataStat('/data/hyx/workspace/CQS/dataset/SQuAD_k/dev-v2.0-k.json')
    # datastat.show_stat()
    file_path = '/data/hyx/workspace/CQS/dataset/SQuAD/train-v2.0.json'
    datastat(file_path)
    file_path = '/data/hyx/workspace/CQS/dataset/SQuAD/dev-v2.0.json'
    datastat(file_path)
