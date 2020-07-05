import jieba

import HyperParameters

class ChatterBot():
    def __init__(self):
        self.temp_speaker_1_datas = []
        self.temp_speaker_2_datas = []

    def loadingData(self):

        with open(HyperParameters.CHATTERBOT_FILE_PATH, encoding='UTF-8') as chatterbot_file:
            lines = chatterbot_file.readlines()
            for line in lines:
                line.rstrip('\n')
                t_index = str(line).find('\t')
                line_speaker_1 = line[0:t_index]
                line_speaker_2 = line[t_index + 1:]

                line_speaker_1_appending = []
                line_speaker_1_cut = jieba.cut(line_speaker_1)
                for char_1 in line_speaker_1_cut:
                    line_speaker_1_appending.append(char_1)
                self.temp_speaker_1_datas.append(line_speaker_1_appending)

                line_speaker_2_appending = []
                line_speaker_2_cut = jieba.cut(line_speaker_2)
                for char_2 in line_speaker_2_cut:
                    line_speaker_2_appending.append(char_2)
                self.temp_speaker_2_datas.append(line_speaker_2_appending)

            self.temp_speaker_1_datas *= 64
            self.temp_speaker_2_datas *= 64

        return self.temp_speaker_1_datas, self.temp_speaker_2_datas