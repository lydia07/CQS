from bert import Ner
import pprint
import json
import tqdm
from tqdm import tqdm
import nltk
from copy import deepcopy
from collections import defaultdict
import re
import unicodedata


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "unknown"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


class SquadData:
    def __init__(self, ner_model, data_dir):
        self.ner_model = ner_model
        self.data_dir = data_dir
        # self.output_dir = output_dir
        self.paragraphs = self.load_data()
        # self.span_len = span_len
        
    def load_data(self):
        with open(self.data_dir, 'r') as f:
            load_dict = json.load(f)
            data = load_dict['data']
            return data

    def generate_keywords(self, sent):
        keywords = []
        word_tag_pairs = self.ner_model.predict(sent)
        keyword = ''
        for pair in word_tag_pairs:
            # print(pair['tag'])
            if pair['tag'][0] == 'B':
                if keyword:
                    keywords.append(keyword)
                keyword = ''
                keyword += pair['word']
            elif pair['tag'][0] == 'I':
                keyword += ' '+pair['word']
        if keyword:
            keywords.append(keyword)
        # print('sent:{},keywords:{}'.format(sent, keywords))
        return keywords

    def process_data(self):
        # counter is used to count the freq of word
        counter = defaultdict(lambda: 0)
        pks = []
        total = 0
        for title in tqdm(self.paragraphs):
            for para in  tqdm(title['paragraphs']):
                context = clean_str(para['context'])
                context_tokens = word_tokenize(context)
                for token in context_tokens:
                    counter[token] += 1
                # spans = convert_idx(context, context_tokens)

                for qa in para['qas']:
                    if qa['is_impossible']:
                        continue
                    total += 1
                    question = clean_str(qa['question'])
                    question_tokens = word_tokenize(question)

                    for token in question_tokens:
                        counter[token] += 1
                    
                    keywords = self.generate_keywords(question)
                    
                    if not keywords:
                        # if keyword is not detected, use answer instead
                        # keywords.append(clean_str(qa['answers'][0]['text'])) 
                        answer = clean_str(qa['answers'][0]['text'])
                        keywords = self.generate_keywords(answer)
                    if not keywords:
                        continue
                    for keyword in keywords:
                        answer = clean_str(qa['answers'][0]['text'])
                        answer_spans = get_spans(context_tokens, answer)
                        keyword_spans = get_spans(context_tokens, keyword)
                        if not (answer_spans and keyword_spans):
                            continue
                        answer_span = answer_spans[0]
                        # if not keyword_spans:
                        #     continue
                        keyword_span = find_closest_span(answer_span, keyword_spans)
                        
                        pk = {
                            'context_tokens': context_tokens,
                            'question_tokens': question_tokens,
                            'start': keyword_span['start'],
                            'end': keyword_span['end'],
                            'keyword': keyword
                        }
                        pks.append(pk)
        return pks, counter

                    
    def make_tag_format(self, pks,src_file, tgt_file):
        src = open(src_file, 'w')
        tgt = open(tgt_file, 'w')
        for para in tqdm(pks):
            c_tokens = para['context_tokens']
            if '\n' in c_tokens:
                print(c_tokens)
                print('new line')
            copied_tokens = deepcopy(c_tokens)
            q_tokens = para['question_tokens']
            # add tag to keywords
            start = para['start']
            end = para['end']

            for idx in range(start, end + 1):
                token = copied_tokens[idx]
                if idx == start:
                    tag = 'B-keyword'
                else:
                    tag = 'I-keyword'
                copied_tokens[idx] = token + '\t' + tag
            
            for token in copied_tokens:
                if '\t' in token:
                    src.write(token + '\n')
                else:
                    src.write(token + "\t" + "O" + "\n")
            
            src.write('\n')
            question = ' '.join(q_tokens)
            tgt.write(question + '\n')
        
        src.close()
        tgt.close()


def find_closest_span(tgtspan, spans):
    res = spans[0]
    min_distance = abs(tgtspan['start'] - res['start'])
    for span in spans:
        distance = abs(tgtspan['start'] - span['start'])
        if distance < min_distance:
            res = span
    return span


def get_spans(tokens, text):
    spans = []
    text = text.rstrip('.')
    text_tokens = word_tokenize(text)
    # print(tokens)
    # print(text_tokens)
    match = False
    for idx, token in enumerate(tokens):
        if token.startswith(text_tokens[0]) or token == text_tokens[0] or token.endswith(text_tokens[0]):
        # if token in text_tokens[0] or text_tokens[0] in token:
            # print(idx)
            i = idx
            for t_token in text_tokens:
                # print('text_token:{}, span_token:{}'.format(tokens[i], t_token))
                # if t_token == '.':
                #     continue
                if i == len(tokens):
                    break
                # if '.' in tokens[i] or '.' in t_token:
                #     print(idx)
                #     print('text_token:{}, span_token:{}'.format(tokens[i], t_token))
                if (tokens[i] not in t_token) and (t_token not in tokens[i]):
                # if not(tokens[i].startswith(t_token) or t_token.startswith(tokens[i])):
                    match = False
                    break
                else:
                    match = True
                    i += 1
            if match:
                span = {}
                span['start'] = idx
                span['end'] = idx + len(text_tokens) - 1
                spans.append(span)
    # if not spans:
    #     print('why???')
    return spans


def make_vocab(counter, vocab_file, max_vocab_size):
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    word2idx = {}
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3

    for idx, (token, freq) in enumerate(sorted_vocab, start=4):
        if len(word2idx) == max_vocab_size:
            break
        word2idx[token] = idx
    with open(vocab_file, "wb") as f:
        pickle.dump(word2idx, f)
        print('vocab file saved')

    return word2idx 


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def clean_str(text):
    text = str(unicodedata.normalize('NFKD', text).encode('ascii','ignore'), encoding='utf-8')
    return text.replace("''", '" ').replace("``", '" ').lower()


def make_embedding(embedding_file, output_file, word2idx):
    word2embedding = dict()
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    num_oov = 0
    # flag = True
    for word, idx in word2idx.items():
        if idx==len(word2idx):
            print('word:{},idx:{}'.format(word, idx))
            break
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            # embedding[idx] = word2embedding['UNK']
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
        print('embedding file saved')
    return embedding
