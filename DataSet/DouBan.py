import jieba
import numpy as np
import HyperParameters

class DouBan():
    def __init__(self):
        self.temp_speaker_1_datas = []
        self.temp_speaker_2_datas = []

    def loadingData(self):

        with open(HyperParameters.DOUBAN_FILE_PATH, encoding='UTF-8') as douban_file:
            lines = douban_file.readlines()
            for line in lines:
                line.rstrip('\n')
                t_index = str(line).find('\t')
                line_speaker_1 = line[0:t_index]
                line_speaker_2 = line[t_index + 1:]
                if "帖子" in line_speaker_1 \
                        or "楼主" in line_speaker_1 \
                        or "楼上" in line_speaker_1 \
                        or "你们" in line_speaker_1 \
                        or "大家" in line_speaker_1 \
                        or "有没有人" in line_speaker_1:
                    line_speaker_1 = line_speaker_1.replace("楼主", "你")
                    line_speaker_1 = line_speaker_1.replace("楼上", "你")
                    line_speaker_1 = line_speaker_1.replace("你们", "你")
                    line_speaker_1 = line_speaker_1.replace("有没有人", "你")
                    line_speaker_1 = line_speaker_1.replace("大家", "你")

                if "帖子" in line_speaker_2 \
                        or "楼主" in line_speaker_2 \
                        or "楼上" in line_speaker_2 \
                        or "有没有人" in line_speaker_2 \
                        or "你们" in line_speaker_2:
                    line_speaker_2 = line_speaker_2.replace("楼主", "你")
                    line_speaker_2 = line_speaker_2.replace("楼上", "你")
                    line_speaker_2 = line_speaker_2.replace("你们", "你")
                    line_speaker_2 = line_speaker_2.replace("有没有人", "你")

                line_speaker_1_refactoring = ''
                line_speaker_1_appending = []
                for char_1 in str(line_speaker_1).split(' '):
                    line_speaker_1_refactoring += char_1
                line_speaker_1_cut = jieba.cut(line_speaker_1_refactoring)
                for char_1 in line_speaker_1_cut:
                    line_speaker_1_appending.append(char_1)

                line_speaker_2_refactoring = ''
                line_speaker_2_appending = []
                for char_2 in str(line_speaker_2).split(' '):
                    line_speaker_2_refactoring += char_2
                line_speaker_2_cut = jieba.cut(line_speaker_2_refactoring)
                for char_2 in line_speaker_2_cut:
                    line_speaker_2_appending.append(char_2)

                self.temp_speaker_1_datas.append(line_speaker_1_appending)
                self.temp_speaker_2_datas.append(line_speaker_2_appending)

            row = np.random.randint(0, len(self.temp_speaker_1_datas), 100000)

        return self.temp_speaker_1_datas[0:], self.temp_speaker_2_datas, row