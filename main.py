import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.generator_model import GeneratorModel
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

    except:
        print("missing or invalid arguments")
        print("usage: python3 main.py -m train|test")
        exit(0)

    if mode == 'train':
        config_file = "configs/config.json"
        config = process_config(config_file)
        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])

        # create your data generator
        data = DataGenerator(config)
        # create an instance of the model you want
        model = GeneratorModel(config, data.word_num)
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
        config_file = "configs/test_config.json"
        config = process_config(config_file)

        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])

        # create your data generator
        data = DataGenerator(config)
        # create an instance of the model you want
        model = GeneratorModel(config, data.word_num)
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
