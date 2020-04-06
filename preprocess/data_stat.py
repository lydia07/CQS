import json


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


if __name__ == '__main__':
    # datastat = DataStat('/data/hyx/workspace/CQS/dataset/SQuAD_k/dev-v2.0-k.json')
    # datastat.show_stat()
    datastat = DataStat('/data/hyx/workspace/CQS/dataset/SQuAD_a/train-v2.0-k.json')
    datastat.show_stat()
