import argparse
import re

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import BertTokenizer


def preprocess(data_path):
    print(f'Load data from {data_path} ...')
    with open(data_path) as f:
        lines = f.readlines()
    sents = sum([line.split('__eou__')[1:-1] for line in lines], [])
    print(f'{len(sents)} sentences.')

    def _clean(s):
        s = s.strip().lower()
        s = re.sub(r'(\w)\.(\w)', r'\1 . \2', s)
        s = s.replace('。', '.')
        s = s.replace(';', ',')
        s = s.replace('’', "'")
        s = s.replace(' p . m . ', ' pm ')
        s = s.replace(' a . m . ', ' am ')
        return s

    print('Clean ...')
    sents = list(map(_clean, sents))
    return sents


def calc_tfidf(sents, tokenizer):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_df=0.9)
    print('Calculate tfidf ...')
    tfidf = tfidf_vectorizer.fit_transform(sents)
    print(f'Calculate done. '
          f'{tfidf.shape[0]} sents, {tfidf.shape[1]} tokens.')
    vocab = {idx: token for token, idx in tfidf_vectorizer.vocabulary_.items()}
    return tfidf, vocab


def get_keywords(idx, sent, tokenizer, tfidf, vocab, ratio):
    # keywords_all = []
    # for i, sent in enumerate(tqdm(sents, desc='Extract Keywords')):
    i = idx
    sent_tokens = tokenizer(sent)
    keywords = []
    for token_id in tfidf[i].nonzero()[1]:
        token = vocab[token_id]
        if token in sent_tokens:
            sent_tokens_id = sent_tokens.index(vocab[token_id])
        else:
            continue
        keywords.append((
            token_id,
            token,
            tfidf[i, token_id],
            sent_tokens_id
        ))
    keywords.sort(key=lambda t: t[2], reverse=True)
    num = int(len(sent_tokens) * ratio)
    num = len(sent_tokens) if num == 0 else num
    keywords = keywords[:num]
    keywords.sort(key=lambda t: t[3])
    keywords = [t[1] for t in keywords]
        # keywords_all.append(keywords)
    return keywords


def save_keywords(keywords, path):
    lines = [' '.join(line) for line in keywords]
    data = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', choices=['train', 'valid', 'test'], required=True)
    parser.add_argument('--ratio', '-r', type=float, default=0.3)
    parser.add_argument('--kw_path', '-k', default=None)
    args = parser.parse_args()

    paths = {
        'train': {
            'data_path': 'data/dialogues_train.txt',
            'keywords_path': 'data/keywords_train.txt',
        },
        'valid': {
            'data_path': 'data/dialogues_validation.txt',
            'keywords_path': 'data/keywords_validation.txt',
        },
        'test': {
            'data_path': 'data/dialogues_test.txt',
            'keywords_path': 'data/keywords_test.txt',
        }
    }

    tokenizer = BertTokenizer('pretrain/vocab.txt')

    sents = preprocess(data_path=paths[args.split]['data_path'])
    tfidf, vocab = calc_tfidf(sents, tokenizer.tokenize)
    keywords = get_keywords(sents, tokenizer.tokenize, tfidf, vocab, ratio=args.ratio)
    kw_path = paths[args.split]['keywords_path'] \
        if args.kw_path is None else args.kw_path
    save_keywords(keywords, kw_path)