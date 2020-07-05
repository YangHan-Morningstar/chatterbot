import numpy as np
import jieba
import json

from DataSet import DouBan, ChatterBot, WebQA, QingYun
import HyperParameters

class DataManager():

    def __init__(self):

        self.encodeTokens, self.decodeTokens, self.outputTokens = [], [], []
        self.sourceTokenDict, self.targetTokenDict = {}, {}
        self.target_token_dict_inv = {}
        self.sourceMaxLen, self.targetMaxLen = 0, 0
        self.testEncodeTokens, self.testDecodeTokens, self.testOutputTokens = [], [], []

        self.initLoadingData = self.readingDataFromJsonFile(HyperParameters.INIT_LOADING_DATA)

        self.douBan = DouBan.DouBan()
        self.chatterBot = ChatterBot.ChatterBot()
        self.webQA = WebQA.WebQA()
        self.qingYun = QingYun.QingYun()

        self.answerSentence = [
            '抱歉，我没有听懂',
            '不好意思，请再说一次',
            '对不起，我还没有学会回答这种问题',
        ]

        if self.initLoadingData == "1":
            self.source_tokens, self.target_tokens = self.loadingTrainingDataFromLocation()
            self.test_source_tokens, self.test_target_tokens = self.loadingTestDataFromLocation()

            self.gettingDict()
            self.selfAddingStartAndEndToken()
            self.gettingMaxLen()

            self.testEncodeTokens, self.testDecodeTokens, self.testOutputTokens = self.padding(
                self.testEncodeTokens,
                self.testDecodeTokens,
                self.testOutputTokens)

            self.testEncodeTokens, self.testDecodeTokens, self.testOutputTokens = self.turningToNum(
                self.testEncodeTokens,
                self.testDecodeTokens,
                self.testOutputTokens)

            self.doingWritingDataToJsonFile()

            with open('/home/tony/AI/Catalina/ListData/initLoadingData.json', 'w') as jsonFile:
                json.dump("0", jsonFile)

    def loadingTrainingDataFromLocation(self):
        speakerOneTrainingDatas = []
        speakerTwoTrainingDatas = []

        if HyperParameters.USING_DOUBAN:
            print("Loading data of douban")
            tempSpeakerOneDatas, tempSpeakerTwoDatas, row = self.douBan.loadingData()
            for i in row:
                speakerOneTrainingDatas.append(tempSpeakerOneDatas[i])
                speakerTwoTrainingDatas.append(tempSpeakerTwoDatas[i])
            print("Data of douban has been successfully loaded")

        if HyperParameters.USING_CHATTERBOT:
            print("Loading data of chatterbot")
            tempSpeakerOneDatas, tempSpeakerTwoDatas = self.chatterBot.loadingData()
            for i in range(len(tempSpeakerOneDatas)):
                speakerOneTrainingDatas.append(tempSpeakerOneDatas[i])
                speakerTwoTrainingDatas.append(tempSpeakerTwoDatas[i])
            print("Data of chatterbot has been successfully loaded")

        if HyperParameters.USING_WEBQA:
            print("Loading data of WebQA(training)")
            tempSpeakerOneDatas, tempSpeakerTwoDatas = self.webQA.loadingTrainingData()
            for i in range(len(tempSpeakerOneDatas)):
                speakerOneTrainingDatas.append(tempSpeakerOneDatas[i])
                speakerTwoTrainingDatas.append(tempSpeakerTwoDatas[i])
            print("Data of WebQA(training) has been successfully loaded")

        if HyperParameters.USING_QINGYUN:
            print("Loading data of qingyun")
            tempSpeakerOneDatas, tempSpeakerTwoDatas = self.qingYun.loadingData()
            for i in range(len(tempSpeakerOneDatas)):
                speakerOneTrainingDatas.append(tempSpeakerOneDatas[i])
                speakerTwoTrainingDatas.append(tempSpeakerTwoDatas[i])
            print("Data of qingyun has been successfully loaded.")

        return speakerOneTrainingDatas, speakerTwoTrainingDatas

    def loadingTestDataFromLocation(self):
        print("Loading data of WebQA(testing)")
        speakerOneTestingDatas, speakerTwoTestingDatas = self.webQA.loadingTestData()
        print("Data of WebQA(testing) has been successfully loaded")

        return speakerOneTestingDatas, speakerTwoTestingDatas

    def dataGenerator(self):
        while True:
            row = np.random.randint(0, len(self.encodeTokens), HyperParameters.BATCH_SIZE)
            encodeTokensX, decodeTokensX, outputTokensY = [], [], []

            for i in row:
                encodeTokensX.append(self.encodeTokens[i])
                decodeTokensX.append(self.decodeTokens[i])
                outputTokensY.append(self.outputTokens[i])

            encodeTokensX, decodeTokensX, outputTokensY = self.padding(
                encodeTokensX,
                decodeTokensX,
                outputTokensY)

            encodeInputX, decodeInputX, decodeOutputY = self.turningToNum(
                encodeTokensX,
                decodeTokensX,
                outputTokensY)

            encodeInputX = np.array(encodeInputX)
            decodeInputX = np.array(decodeInputX)
            decodeOutputY = np.array(decodeOutputY)

            yield [encodeInputX, decodeInputX], decodeOutputY


    def gettingDict(self):
        self.sourceTokenDict = self.buildingTokensDict(self.source_tokens, self.test_source_tokens)
        self.targetTokenDict = self.buildingTokensDict(self.target_tokens, self.test_target_tokens)
        self.target_token_dict_inv = {v: k for k, v in self.targetTokenDict.items()}

    def selfAddingStartAndEndToken(self):
        self.encodeTokens = [['<START>'] + tokens + ['<END>'] for tokens in self.source_tokens]
        self.decodeTokens = [['<START>'] + tokens + ['<END>'] for tokens in self.target_tokens]
        self.outputTokens = [tokens + ['<END>', '<PAD>'] for tokens in self.target_tokens]
        self.testEncodeTokens = [['<START>'] + tokens + ['<END>'] for tokens in self.test_source_tokens]
        self.testDecodeTokens = [['<START>'] + tokens + ['<END>'] for tokens in self.test_target_tokens]
        self.testOutputTokens = [tokens + ['<END>', '<PAD>'] for tokens in self.test_target_tokens]

    def addingStartAndEndToken(self, source_tokens, target_tokens):
        encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
        decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
        output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

        return encode_tokens, decode_tokens, output_tokens


    def gettingMaxLen(self):
        source_max_len = max(map(len, self.encodeTokens))
        target_max_len = max(map(len, self.decodeTokens))
        test_source_max_len = max(map(len, self.testEncodeTokens))
        test_target_max_len = max(map(len, self.testDecodeTokens))

        self.sourceMaxLen = max(source_max_len, test_source_max_len)
        self.targetMaxLen = max(target_max_len, test_target_max_len)

    def padding(self, encode_tokens_i, decode_tokens_i, output_tokens_i):
        encode_tokens_i = [tokens + ['<PAD>'] * (self.sourceMaxLen - len(tokens)) for tokens in encode_tokens_i]
        decode_tokens_i = [tokens + ['<PAD>'] * (self.targetMaxLen - len(tokens)) for tokens in decode_tokens_i]
        output_tokens_i = [tokens + ['<PAD>'] * (self.targetMaxLen - len(tokens)) for tokens in output_tokens_i]
        return encode_tokens_i, decode_tokens_i, output_tokens_i

    def turningToNum(self, encode_tokens_i, decode_tokens_i, output_tokens_i):
        encode_input_i = [list(map(lambda x: self.sourceTokenDict[x], tokens)) for tokens in encode_tokens_i]
        decode_input_i = [list(map(lambda x: self.targetTokenDict[x], tokens)) for tokens in decode_tokens_i]
        decode_output_i = [list(map(lambda x: [self.targetTokenDict[x]], tokens)) for tokens in output_tokens_i]
        return encode_input_i, decode_input_i, decode_output_i

    def doingWritingDataToJsonFile(self):
        self.writingDataToJsonFile(HyperParameters.ENCODE_TOKENS_FILE_PATH, self.encodeTokens)
        self.writingDataToJsonFile(HyperParameters.DECODE_TOKENS_FILE_PATH, self.decodeTokens)
        self.writingDataToJsonFile(HyperParameters.OUTPUT_TOKENS_FILE_PATH, self.outputTokens)
        self.writingDataToJsonFile(HyperParameters.SOURCE_TOKEN_DICT_FILE_PATH, self.sourceTokenDict)
        self.writingDataToJsonFile(HyperParameters.TARGET_TOKEN_DICT_FILE_PATH, self.targetTokenDict)
        self.writingDataToJsonFile(HyperParameters.SOURCE_MAX_LEN_FILE_PATH, self.sourceMaxLen)
        self.writingDataToJsonFile(HyperParameters.TARGET_MAX_LEN_FILE_PATH, self.targetMaxLen)
        self.writingDataToJsonFile(HyperParameters.TARGET_TOKEN_DICT_INV_FILE_PATH, self.target_token_dict_inv)
        self.writingDataToJsonFile(HyperParameters.TEST_ENCODE_TOKENS_FILE_PATH, self.testEncodeTokens)
        self.writingDataToJsonFile(HyperParameters.TEST_DECODE_TOKENS_FILE_PATH, self.testDecodeTokens)
        self.writingDataToJsonFile(HyperParameters.TEST_OUTPUT_TOKENS_FILE_PATH, self.testOutputTokens)

    def loadingProcessedDataFromLocation(self):
        encode_tokens = self.readingDataFromJsonFile(HyperParameters.ENCODE_TOKENS_FILE_PATH)
        decode_tokens = self.readingDataFromJsonFile(HyperParameters.DECODE_TOKENS_FILE_PATH)
        output_tokens = self.readingDataFromJsonFile(HyperParameters.OUTPUT_TOKENS_FILE_PATH)
        test_encode_tokens = self.readingDataFromJsonFile(HyperParameters.TEST_ENCODE_TOKENS_FILE_PATH)
        test_decode_tokens = self.readingDataFromJsonFile(HyperParameters.TEST_DECODE_TOKENS_FILE_PATH)
        test_output_tokens = self.readingDataFromJsonFile(HyperParameters.TEST_OUTPUT_TOKENS_FILE_PATH)
        source_token_dict = self.readingDataFromJsonFile(HyperParameters.SOURCE_TOKEN_DICT_FILE_PATH)
        target_token_dict = self.readingDataFromJsonFile(HyperParameters.TARGET_TOKEN_DICT_FILE_PATH)
        source_max_len = self.readingDataFromJsonFile(HyperParameters.SOURCE_MAX_LEN_FILE_PATH)
        target_max_len = self.readingDataFromJsonFile(HyperParameters.TARGET_MAX_LEN_FILE_PATH)
        target_token_dict_inv = self.readingDataFromJsonFile(HyperParameters.TARGET_TOKEN_DICT_INV_FILE_PATH)
        return encode_tokens, decode_tokens, output_tokens, test_encode_tokens, test_decode_tokens, test_output_tokens, source_token_dict, target_token_dict, source_max_len, target_max_len, target_token_dict_inv

    def readingDataFromJsonFile(self, file_path):
        with open(file_path) as file:
            file_data = json.load(file)
        return file_data

    def writingDataToJsonFile(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def buildingTokensDict(self, token_list_train, token_list_test=[]):
        token_dict = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
        }
        for tokens in token_list_train:
            for token in tokens:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)

        for tokens in token_list_test:
            for token in tokens:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)

        return token_dict

    def cutting_sentence(self, sentence):
        sentence_return = []
        sentence_cutted = jieba.cut(sentence)
        for character in sentence_cutted:
            sentence_return.append(character)
        return sentence_return


