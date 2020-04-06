import json
from nltk.tokenize import sent_tokenize


class DataGenerator:
    def __init__(self, input_path, output_path):
        self.input_file = input_path
        self.output_file = output_path
        self.data = self.load_data()

    def load_data(self):
        with open(self.input_file, 'r') as f:
            load_dict = json.load(f)
            data = load_dict['data']
            return data
                    

    # def format(self, sent):
    #     res = sent.replace
    #     res = sent.replace(')', '-rrb-')
    #     return res

    def get_sentences(self, text):
        sent_tokenize_list = sent_tokenize(text)
        return sent_tokenize_list

    def get_formatted_json(self):
        res = []
        is_first_line = True
        with open(self.output_file, 'a') as f:
            data = self.load_data()
            for data_id, title in enumerate(data):
                for para_id, p in enumerate(title['paragraphs']):
                    sents = self.get_sentences(p['context'])
                    for sent in sents:
                        json_data = {}
                        # json_data['src'] = self.format(sent)
                        json_data['src'] = sent
                        json_data['tgt'] = ''
                        json_data['data_id'] = data_id
                        json_data['para_id'] = para_id
                        json_data = json.dumps(json_data)
                        if not is_first_line:
                            f.write('\n')
                        else:
                            is_first_line = False
                        f.write(json_data)
                        res.append(json_data)
            return res
        
if __name__ == '__main__':
    input_path = '/data/hyx/workspace/CQS/dataset/SQuAD/train-v2.0.json'
    output_path = '/data/hyx/workspace/CQS/dataset/SQuAD_a/train.json'
    data_generator = DataGenerator(input_path, output_path)
    data_generator.get_formatted_json()



