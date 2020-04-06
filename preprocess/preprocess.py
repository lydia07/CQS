from bert import Ner
import pprint
import json
import tqdm
from tqdm import tqdm

class KeywordGenerator:

    def __init__(self, data_dir=None, output_dir=None, enhance=False, src_path=None, tgt_path=None, ner_model=None):
        self.ner_model = ner_model
        self.output_dir = output_dir
        self.enhance = enhance
        self.src_path = src_path
        self.tgt_path = tgt_path
        if data_dir and output_dir:
            self.paragraphs = self.load_data(data_dir)
        

    def load_data(self, data_dir):
        paragraphs = []
        with open(data_dir, 'r') as f:
            load_dict = json.load(f)
            data = load_dict['data']
            if self.enhance:
                data = self.enhance_data(data, self.src_path, self.tgt_path)
            for title in data:
                for p in title['paragraphs']:
                    paragraph = {}
                    paragraph['context'] = p['context']
                    questions = []
                    for q in p['qas']:
                        # including impossible questions
                        if not q['is_impossible']:
                            questions.append(q['question'])
                    paragraph['questions'] = questions
                    paragraphs.append(paragraph)
        # print(paragraphs[0])
        return paragraphs

    def enhance_data(self, data, src_path, tgt_path):
        res = data
        src_file = open(src_path, 'r')
        tgt_file = open(tgt_path, 'r')
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()
        for src, tgt in zip(src_lines, tgt_lines):
            src = json.loads(src)
            data_id = src['data_id']
            para_id = src['para_id']
            generated_q = {}
            generated_q['question'] = tgt
            generated_q['is_impossible'] = True
            res[data_id]['paragraphs'][para_id]['qas'].append(generated_q)
        return res

    def predict(self, sent):
        output = self.ner_model.predict(sent)
        return output

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
        print('sent:{},keywords:{}'.format(sent, keywords))
        return keywords

    def add_keywords(self):
        paragraphs_kw = []
        for p in tqdm(self.paragraphs):
            paragraph_kw = {}
            paragraph_kw['context'] = p['context']
            paragraph_kw['keywords'] = {}
            for q in p['questions']:
                keywords = self.generate_keywords(q)
                for k in keywords:
                    if k not in paragraph_kw['keywords'].keys():
                        paragraph_kw['keywords'][k] = []
                    paragraph_kw['keywords'][k].append(q)
            print(paragraph_kw)
            if len(paragraph_kw['keywords'].keys()):
                paragraphs_kw.append(paragraph_kw)
        return paragraphs_kw

    def save_data(self):
        data = self.add_keywords()
        with open(self.output_dir, 'w') as f:
            data = json.dumps(data)
            f.write(data)


if __name__ == '__main__':
    ner_model = Ner('/data/hyx/workspace/BERT-NER/out_base/')
    # keyword_generator = KeywordGenerator(ner_model=ner_model)
    # result = keyword_generator.predict("What was the name Loughborough University was known by from 1966 to 1996?")
    # print(result)
    # res = keyword_generator.generate_keywords("What was the name Loughborough University was known by from 1966 to 1996?")
    # print(res)
    data_dir='/data/hyx/workspace/CQS/dataset/SQuAD/train-v2.0.json'
    output_dir='/data/hyx/workspace/CQS/dataset/SQuAD_a/train-v2.0-k.json'
    src_path = '/data/hyx/workspace/CQS/dataset/SQuAD_a/train.json'
    tgt_path = '/data/hyx/workspace/CQS/dataset/SQuAD_a/output.json'
    enhance = True
    keyword_generator = KeywordGenerator(ner_model=ner_model, data_dir=data_dir, output_dir=output_dir, enhance=enhance, src_path=src_path, tgt_path=tgt_path)
    keyword_generator.save_data()
