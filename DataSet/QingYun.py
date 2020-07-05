import jieba

import HyperParameters

class QingYun():
    def __init__(self):
        self.temp_speaker_1_datas = []
        self.temp_speaker_2_datas = []

    def loadingData(self):

        with open(HyperParameters.QINGYUN_FILE_PATH, encoding='UTF-8') as qingyun_file:
            lines = qingyun_file.readlines()
            for line in lines:
                line.rstrip('\n')
                t_index = str(line).find('|')
                line_speaker_1 = line[0:t_index - 1]
                line_speaker_2 = line[t_index + 2:]

                if "*" in line_speaker_1:
                    break

                line_speaker_1_appending = []
                line_speaker_1_cut = jieba.cut(line_speaker_1)
                for char_1 in line_speaker_1_cut:
                    line_speaker_1_appending.append(char_1)
                line_speaker_2_appending = []
                line_speaker_2_cut = jieba.cut(line_speaker_2)
                for char_2 in line_speaker_2_cut:
                    line_speaker_2_appending.append(char_2)

                self.temp_speaker_1_datas.append(line_speaker_1_appending)
                self.temp_speaker_2_datas.append(line_speaker_2_appending)

            self.temp_speaker_1_datas *= 64
            self.temp_speaker_2_datas *= 64

        return self.temp_speaker_1_datas, self.temp_speaker_2_datas


