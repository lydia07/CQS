import json
import pickle
import pprint
import sys
sys.path.append("..")
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy

import nltk
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from tqdm import tqdm
# import bert.Ner as Ner
# from bert import Ner
from preprocess.bert import Ner
import config

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNK"
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
        print(counter['beyonce'])
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


class SquadDataWithTag(data.Dataset):
    def __init__(self, src_file, tgt_file, max_len, word2idx):
        self.srcs = []
        self.tags = []

        lines = open(src_file, 'r').readlines()
        paragraph, tags = [], []
        self.entity2idx = {
                            'O': 0,
                            'B-keyword': 1,
                            'I-keyword': 2
                            }
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                paragraph.insert(0, START_TOKEN)
                paragraph.append(END_TOKEN)
                self.srcs.append(paragraph)

                tags.insert(0, self.entity2idx['O'])
                tags.append(self.entity2idx['O'])
                self.tags.append(tags)
                assert len(paragraph) == len(tags)
                paragraph, tags = [], []
            else:
                tokens = line.split('\t')
                word, tag = tokens[0], tokens[1]
                paragraph.append(word)
                tags.append(self.entity2idx[tag])
            
        self.tgts = open(tgt_file, 'r').readlines()

        assert len(self.srcs) == len(self.tgts),\
            "the number of source sequence {}" " and target sequence {} must be the same" \
                .format(len(self.srcs), len(self.trgs))
        
        self.max_len = max_len
        self.word2idx = word2idx
        self.num_seqs = len(self.srcs)

    def __getitem__(self, index):
        src = self.srcs[index]
        tgt = self.tgts[index]
        tag = self.tags[index]

        tag = torch.Tensor(tag[:self.max_len])
        src, ext_src, oov_lst = self.context2ids(src ,self.word2idx)
        tgt, ext_tgt = self.question2ids(tgt, self.word2idx, oov_lst)
        return src, ext_src, tgt, ext_tgt, oov_lst, tag
        
    def __len__(self):
        return self.num_seqs

    def context2ids(self, tokens, word2idx):
        ids = []
        ext_ids = []
        oov_lst = []
        # START and END token is already in tokens lst
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                ext_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                ext_ids.append(len(word2idx) + oov_lst.index(token))
            if len(ids) == self.max_len:
                break
        
        ids = torch.Tensor(ids)
        ext_ids = torch.Tensor(ext_ids)
        return ids, ext_ids, oov_lst

    def question2ids(self, seq, word2idx, oov_lst):
        ids = []
        ext_ids = []
        ids.append(word2idx[START_TOKEN])
        ext_ids.append(word2idx[START_TOKEN])
        tokens = seq.strip().split(' ')

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                ext_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token in oov_lst:
                    ext_ids.append(len(word2idx) + oov_lst.index(token))
                else:
                    ext_ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])
        ext_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        ext_ids = torch.Tensor(ext_ids)

        return ids, ext_ids


def get_loader(src_file, tgt_file, word2idx, batch_size):
        dataset = SquadDataWithTag(src_file, tgt_file, config.max_seq_len,
                                   word2idx)
        dataloader = data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=collate_fn_tag)
        return dataloader


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
    # print('len_word2idx:{}'.format(len(word2idx)))
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
    unk_vec = np.random.rand(300)
    print(unk_vec)
    word2embedding[UNK_TOKEN] = unk_vec
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    num_oov = 0
    # flag = True
    for word, idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            # embedding[idx] = word2embedding['unknown']
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
        print('embedding file saved')
    return embedding


def collate_fn_tag(data):
    def merge(sequences):
        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst, tag_seqs = zip(*data)

    src_seqs, src_len = merge(src_seqs)
    ext_src_seqs, _ = merge(ext_src_seqs)
    trg_seqs, trg_len = merge(trg_seqs)
    ext_trg_seqs, _ = merge(ext_trg_seqs)
    tag_seqs, _ = merge(tag_seqs)

    assert src_seqs.size(1) == tag_seqs.size(1), "length of tokens and tags should be equal"

    return src_seqs, ext_src_seqs, src_len, trg_seqs, ext_trg_seqs, trg_len, tag_seqs, oov_lst
