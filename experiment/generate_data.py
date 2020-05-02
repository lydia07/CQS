import sys
sys.path.append('..')
import config
import codecs
import json
from preprocess.keywords import (calc_tfidf, get_keywords)
from preprocess.data_utils import (merge_keywords, clean_str, word_tokenize)

def load_queries(file_path):
    queries = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines[: 100000]:
            id, query = line.split('\t')
            query = clean_str(query)
            queries.append(query)
    # print(len(queries))
    return queries

def generate_data(sents, output_file, ratio=0.5, criterion=0.35):
    f = open(output_file, 'w')
    res = {}
    res['data'] = []
    tfidf, vocab = calc_tfidf(sents, word_tokenize)
    cnt = 0
    for idx, query in enumerate(sents):
        data = {}
        query = query.strip('\n').strip('\r')
        keywords = get_keywords(idx, query, word_tokenize, tfidf, vocab, ratio)
        keywords = merge_keywords(query, keywords)
        length, keyword = max([(len(x), x) for x in keywords])
        pct = length / len(query)
        if pct >= criterion:
            cnt += 1
            data['query'] = query
            data['keyword'] = keyword
            res['data'].append(data)
        # print('query:{}, keyword:{}, %:{:.2}'.format(query[:-1], keyword, length/len(query)))
    res['num'] = cnt
    res = json.dumps(res)
    f.write(res)
    f.close()
    return cnt

if __name__ == "__main__":
    queries = load_queries(config.trec_file)
    generate_data(queries, config.output_file, criterion=config.criterion)