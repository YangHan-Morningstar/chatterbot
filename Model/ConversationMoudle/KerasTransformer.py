from keras_transformer import get_model

from DataManager import DataManager
import HyperParameters


class KerasTransformer():

    def __init__(self):
        self.model = None
        self.dataManager = DataManager.DataManager()

        print("Loading processed data from location.")
        self.encode_tokens, self.decode_tokens, self.output_tokens, self.test_encode_tokens, self.test_decode_tokens, \
        self.test_output_tokens, self.source_token_dict, self.target_token_dict, self.source_max_len, self.target_max_len, \
        self.target_token_dict_inv = self.dataManager.loadingProcessedDataFromLocation()
        print("System has loaded processed data from location successfully.")

        self.assigningToDataManager()

    def getModel(self):
        print("Beginning to build the model.")
        model = get_model(
            token_num=max(len(self.source_token_dict), len(self.target_token_dict)),
            embed_dim=HyperParameters.EMBED_DIM,
            encoder_num=HyperParameters.ENCODER_NUM,
            decoder_num=HyperParameters.DECODER_NUM,
            head_num=HyperParameters.HEAD_NUM,
            hidden_dim=HyperParameters.HIDDEN_DIM,
            dropout_rate=HyperParameters.DROPOUT_RATE,
            use_same_embed=False,
        )
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
        print("The model has been built successfully and the summary of it is")
        model.summary()

        return model

    def trainingModel(self):
        if self.model is None:
            self.model = self.getModel()
        try:
            self.model.fit_generator(
                self.dataManager.dataGenerator(),
                epochs=HyperParameters.EPOCHS,
                steps_per_epoch=len(self.encode_tokens) // (HyperParameters.BATCH_SIZE)
            )
            print('Model has been trained and will be saved.')
        except:
            print("There are some errors and training has been down ahead of time.")
        finally:
            print("The current model has been trained and the parameters of it will be saved.")
            self.model.save_weights(HyperParameters.WEIGHT_SAVE_PATH)
            print("Model has been saved as " + HyperParameters.WEIGHT_SAVE_PATH)

    def retrainingModel(self):
        self.model = self.getModel()
        print("Beginning to load pretrained weights and retrain the model.")
        self.model.load_weights(HyperParameters.RETRAINING_WEIGHT_SAVE_PATH)
        print("Model's weights has been loaded.")
        self.model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
        self.trainingModel()
        self.model = None

    def testingModel(self):
        self.model = self.getModel()
        print("Beginning to load pretrained weights and test the model.")
        self.model.load_weights(HyperParameters.TESTING_WEIGHT_SAVE_PATH)
        print("Model's weights has been loaded.")
        self.model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

    def assigningToDataManager(self):
        self.dataManager.encodeTokens = self.encode_tokens
        self.dataManager.decodeTokens = self.decode_tokens
        self.dataManager.outputTokens = self.output_tokens
        self.dataManager.sourceMaxLen = self.source_max_len
        self.dataManager.targetMaxLen = self.target_max_len
        self.dataManager.sourceTokenDict = self.source_token_dict
        self.dataManager.targetTokenDict = self.target_token_dict
        self.dataManager.target_token_dict_inv = self.target_token_dict_inv
        self.dataManager.testEncodeTokens = self.test_encode_tokens
        self.dataManager.testDecodeTokens = self.test_decode_tokens
        self.dataManager.testOutputTokens = self.test_output_tokens

