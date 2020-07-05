import jieba
import json
import HyperParameters

class WebQA():
    def __init__(self):
        self.temp_speaker_1_training_datas = []
        self.temp_speaker_2_training_datas = []
        self.temp_speaker_1_testing_datas = []
        self.temp_speaker_2_testing_datas = []

    def loadingTrainingData(self):

        with open(HyperParameters.WEBQA_FILE_PATH, encoding='UTF-8') as json_file:
            datas_file = json.load(json_file)
            for key_parent in datas_file:
                line_question, line_evidences = datas_file[str(key_parent)]['question'], \
                                                datas_file[str(key_parent)]['evidences']

                line_speaker_1_appending = []
                sentence_1_cut = jieba.cut(line_question)
                for char_1 in sentence_1_cut:
                    line_speaker_1_appending.append(char_1)
                self.temp_speaker_1_training_datas.append(line_speaker_1_appending)

                for key_child in line_evidences:
                    line_ans = line_evidences[key_child]['answer'][0]
                    if line_ans != 'no_answer':
                        break
                line_ans = ''.join(line_ans)
                if line_ans == 'no_answer':
                    line_ans = '没有查询到答案'

                line_speaker_2_appending = []
                line_speaker_2_cut = jieba.cut(line_ans)
                for char_2 in line_speaker_2_cut:
                    line_speaker_2_appending.append(char_2)
                self.temp_speaker_2_training_datas.append(line_speaker_2_appending)

        return self.temp_speaker_1_training_datas, self.temp_speaker_2_training_datas

    def loadingTestData(self):
        with open(HyperParameters.WEBQA_TEST_FILE_PATH_1, encoding='UTF-8') as json_file:
            datas_file = json.load(json_file)
            for key_parent in datas_file:
                line_question, line_evidences = datas_file[str(key_parent)]['question'], datas_file[str(key_parent)][
                    'evidences']

                line_speaker_1_appending = []
                sentence_1_cut = jieba.cut(line_question)
                for char_1 in sentence_1_cut:
                    line_speaker_1_appending.append(char_1)
                self.temp_speaker_1_testing_datas.append(line_speaker_1_appending)

                for key_child in line_evidences:
                    line_ans = line_evidences[key_child]['answer'][0]
                    if line_ans != 'no_answer':
                        break
                line_ans = ''.join(line_ans)
                if line_ans == 'no_answer':
                    line_ans = '没有查询到答案'

                line_speaker_2_appending = []
                line_speaker_2_cut = jieba.cut(line_ans)
                for char_2 in line_speaker_2_cut:
                    line_speaker_2_appending.append(char_2)
                self.temp_speaker_2_testing_datas.append(line_speaker_2_appending)

        return self.temp_speaker_1_testing_datas, self.temp_speaker_2_testing_datas
