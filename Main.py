from Model.ConversationMoudle import KerasTransformer
from Model.VoiceGeneratorModule import BaiDuVoiveGenerator
from keras_transformer import decode
import numpy as np
import HyperParameters

class Catalina():
    def __init__(self):
        self.loadingCatalinaConversationModule()

        self.trainingModel = HyperParameters.TRAINING_MODEL
        self.retrainingModel = HyperParameters.RETRAINING_MODEL
        self.testingModel = HyperParameters.TESTING_MODEL
        self.counter = 0

    def loadingCatalinaConversationModule(self):
        self.kerasTransformer = KerasTransformer.KerasTransformer()
        self.voiceGenerator = BaiDuVoiveGenerator.BaiDuVoiceGenerator()

    def action(self):

        if self.trainingModel:
            self.kerasTransformer.trainingModel()

        elif self.retrainingModel:
            self.kerasTransformer.retrainingModel()

        elif self.testingModel:
            self.kerasTransformer.testingModel()
            self.speaking()

    def speaking(self):
        while True:
            print("Tony:")
            ans = ''
            target_tokens = ''
            speaker_1_words = input()
            speaker_1_words_cut = self.kerasTransformer.dataManager.cutting_sentence(speaker_1_words)
            source_tokens = [speaker_1_words_cut]

            try:
                encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
                encode_input = [list(map(lambda x: self.kerasTransformer.source_token_dict[x], tokens)) for tokens in encode_tokens]
                decoded = decode(
                    self.kerasTransformer.model,
                    encode_input,
                    start_token=self.kerasTransformer.target_token_dict['<START>'],
                    end_token=self.kerasTransformer.target_token_dict['<END>'],
                    pad_token=self.kerasTransformer.target_token_dict['<PAD>'],
                )
                for index in decoded[0][1:-1]:
                    ans += self.kerasTransformer.target_token_dict_inv[str(index)]
                ans_words_cut = self.kerasTransformer.dataManager.cutting_sentence(ans)
                target_tokens = [ans_words_cut]
            except KeyError:
                row = np.random.randint(0, 3, 1)
                ans = self.kerasTransformer.dataManager.ans_sentence[row[0]]
                print(KeyError)
            finally:
                if ans in self.kerasTransformer.dataManager.answerSentence:
                    print('Catalina:\n' + ans)
                else:
                    try:
                        encode_tokens_x, decode_tokens_x, output_tokens_y = self.kerasTransformer.dataManager.addingStartAndEndToken(
                            source_tokens,
                            target_tokens)

                        encode_tokens_x, decode_tokens_x, output_tokens_y = self.kerasTransformer.dataManager.padding(
                            encode_tokens_x,
                            decode_tokens_x,
                            output_tokens_y)

                        encode_input_x, decode_input_x, decode_output_y = self.kerasTransformer.dataManager.turningToNum(
                            encode_tokens_x,
                            decode_tokens_x,
                            output_tokens_y)

                        encode_input_x = np.array(encode_input_x)
                        decode_input_x = np.array(decode_input_x)
                        decode_output_y = np.array(decode_output_y)

                        loss_value, metric_value = self.kerasTransformer.model.evaluate([encode_input_x, decode_input_x],
                                                                       decode_output_y)
                        if loss_value >= 0.1:
                            row = np.random.randint(0, 3, 1)
                            ans = self.kerasTransformer.dataManager.answerSentence[row[0]]
                    except KeyError:
                        row = np.random.randint(0, 3, 1)
                        ans = self.kerasTransformer.dataManager.ans_sentence[row[0]]
                        print(KeyError)

                    print('Catalina:\n' + ans)

                # self.voiceGenerator.playingSound(ans, self.counter)
                # self.counter += 1



if __name__ == '__main__':

    Cat = Catalina()

    Cat.action()
