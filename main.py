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
        exit(0)

    if mode == 'train':
        config_file = "configs/config.json"
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
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config, logger)

    if mode == 'train':
        # here you train your model
        trainer.train()
    elif mode == 'test':
        trainer.create_poetry()

def test_data():
    config_file = 'configs/config.json'
    config = process_config(config_file)
    generator = DataGenerator(config)
    x, y = next(generator.next_batch(5))
    print(x.shape)
    # print(generator.dictionary[''])
    print(generator.dictionary[' '])

if __name__ == '__main__':
    main()
    # test_data()
