import tensorflow as tf

import _init_paths
from data_loader.data_generator_softmax import IVUSDataGenerator
from utils.logger import Logger
from utils.utils import get_args, create_dirs, process_config

# change this when working on different models
from models.ivusnet import Model
# change this when training with prob masks
from trainers.softmax_model_trainer import SigmoidTrainer


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print(e)
        print('Missing or invalid arguments')
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create instance of the model you want
    model = Model(config)
    # load model if exist
    model.load(sess)
    # create your data generator
    data = IVUSDataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = SigmoidTrainer(sess, model, data, config, logger)

    # train the model
    trainer.train()


if __name__ == '__main__':
    import time

    s = time.time()
    main()
    print('Time used', time.time() - s)
