import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.LSTMModel import LSTMModel
from models.GRUModel import GRUModel
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        mode = args.mode
        cell = args.cell

    except:
        print("missing or invalid arguments")
        print("usage: python3 main.py -m train|test -c lstm|gru")
        exit(0)

    if cell not in ['lstm', 'gru']:
        print("usage: python3 main.py -m train|test -c lstm|gru")
        exit(0)

    if mode == 'train':
        if cell == 'lstm':
            config_file = "configs/config.json"
        else:
            config_file = "configs/gru_config.json"
        config = process_config(config_file)
        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])

        # create your data generator
        data = DataGenerator(config)
        # create an instance of the model you want
        if cell == 'lstm':
            model = LSTMModel(config, data.word_num)
        else:
            model = GRUModel(config, data.word_num)
        # create tensorflow session
        sess = tf.Session()
        #load model if exists
        model.load(sess)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = Trainer(sess, model, data, config, logger)
        trainer.train()
    elif mode == 'test':
        if cell == 'lstm':
            config_file = "configs/test_lstm_config.json"
        else:
            config_file = "configs/test_gru_config.json"
        config = process_config(config_file)

        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])

        # create your data generator
        data = DataGenerator(config)
        # create an instance of the model you want
        if cell == 'lstm':
            model = LSTMModel(config, data.word_num)
        else:
            model = GRUModel(config, data.word_num)
        # create tensorflow session
        sess = tf.Session()
        #load model if exists
        model.load(sess)
        # create tensorboard logger
        logger = Logger(sess, config)
        model.create_poetry(sess, data)


def test_data():
    config_file = 'configs/config.json'
    config = process_config(config_file)
    generator = DataGenerator(config)
    x, y = generator.generateBatch(2)
    print(x[0].shape)
    # print(generator.dictionary[''])
    print(generator.dictionary[' '])
    print(generator.dictionary['['])
    print(generator.dictionary[']'])
    print(generator.dictionary['，'])
    print(generator.dictionary['。'])



if __name__ == '__main__':
    main()
    # test_data()
