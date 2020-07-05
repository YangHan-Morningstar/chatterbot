USING_DOUBAN = True

USING_CHATTERBOT = True

USING_WEBQA = False

USING_QINGYUN = True

TRAINING_MODEL = False

RETRAINING_MODEL = False

TESTING_MODEL = True

BATCH_SIZE = 32

EMBED_DIM = 64

ENCODER_NUM = 2

DECODER_NUM = 2

HEAD_NUM = 4

HIDDEN_DIM = 128

DROPOUT_RATE = 0.05

EPOCHS = 5

SOURCE_MAX_LEN = 500

TARGET_MAX_LEN = 1000

DOUBAN_FILE_PATH = 'DataSet/chinese_chatbot_corpus/douban_single_turn.tsv'

CHATTERBOT_FILE_PATH = 'DataSet/chinese_chatbot_corpus/chatterbot.tsv'

WEBQA_FILE_PATH = 'DataSet/WebQA.v1.0/me_train.json'

WEBQA_TEST_FILE_PATH_1 = 'DataSet/WebQA.v1.0/me_test.ir.json'

WEBQA_TEST_FILE_PATH_2 = 'DataSet/WebQA.v1.0/me_test_ann.json'

QINGYUN_FILE_PATH = 'DataSet/chinese_chatbot_corpus/qingyun.csv'

ENCODE_TOKENS_FILE_PATH = 'ListData/encode_tokens.json'

DECODE_TOKENS_FILE_PATH = 'ListData/decode_tokens.json'

OUTPUT_TOKENS_FILE_PATH = 'ListData/output_tokens.json'

TEST_ENCODE_TOKENS_FILE_PATH = 'ListData/test_encode_tokens.json'

TEST_DECODE_TOKENS_FILE_PATH = 'ListData/test_decode_tokens.json'

TEST_OUTPUT_TOKENS_FILE_PATH = 'ListData/test_output_tokens.json'

SOURCE_TOKEN_DICT_FILE_PATH = 'ListData/source_token_dict.json'

TARGET_TOKEN_DICT_FILE_PATH = 'ListData/target_token_dict.json'

SOURCE_MAX_LEN_FILE_PATH = 'ListData/source_max_len.json'

TARGET_MAX_LEN_FILE_PATH = 'ListData/target_max_len.json'

TARGET_TOKEN_DICT_INV_FILE_PATH = 'ListData/target_token_dict_inv.json'

WEIGHT_SAVE_PATH = 'ModelTrainedParameters/model_1.9.3.hdf5'

RETRAINING_WEIGHT_SAVE_PATH = 'ModelTrainedParameters/model_1.9.1.hdf5'

TESTING_WEIGHT_SAVE_PATH = 'ModelTrainedParameters/model_1.9.hdf5'

INIT_LOADING_DATA = 'ListData/initLoadingData.json'

