from bert import Ner
import pprint
import json
import tqdm
from tqdm import tqdm

class KeywordGenerator:

    def __init__(self, data_dir, output_dir, ner_model=None, base_dir='dataset/SQuAD/'):
        self.ner_model = ner_model
        self.paragraphs = self.load_data(base_dir+data_dir)
        self.output_dir = output_dir

    def load_data(self, data_dir):
        paragraphs = []
        with open(data_dir, 'r') as f:
            load_dict = json.load(f)
            data = load_dict['data']
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

    def predict(self, sent):
        output = self.ner_model.predict(sent)
        return output

    def generate_keywords(self, sent):
        keywords = []
        word_tag_pairs = self.ner_model.predict(sent)
        keyword = ''
        for pair in word_tag_pairs:
            if pair['tag'][0] == 'B':
                if keyword:
                    keywords.append(keyword)
                keyword = ''
                keyword += pair['word']
            elif pair['tag'][0] == 'I':
                keyword += ' '+pair['word']
        if keyword:
            keywords.append(keyword)
        print(keywords)
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
    ner_model = Ner(device=2)
    # keyword_generator = KeywordGenerator(ner_model)
    # result = keyword_generator.predict("In what country is Normandy located?")
    keyword_generator = KeywordGenerator(ner_model=ner_model, data_dir='train-v2.0.json', output_dir='dataset/SQuAD_k/train-v2.0-k.json')
    keyword_generator.save_data()
