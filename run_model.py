import os
import random
import argparse

from pytorch_lightning import seed_everything

from utils import load_json


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Unsupervised Lesion Detection')
    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-i', '--inference', action='store_true')
    parser.add_argument('-e', '--export', action='store_true')
    parser.add_argument('-p', '--savepath', default='./')
    args = parser.parse_args()

    config = load_json(args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices
    seed = config.run.seed or random.randint(1, 10000)
    seed_everything(seed)
    print('Using manual seed: {}'.format(seed))
    print('Config: ', config)

    if args.train:
        print('Starting model training...')

        from trainers import const_trainer_train
        trainer, model = const_trainer_train(config, seed, args)
        trainer.fit(model)

    elif args.inference:
        print('Starting model inference...')

        from trainers import const_trainer_test
        trainer, model = const_trainer_test(config, seed, args)
        trainer.test(model)

    elif args.export:
        print('Exporting saved models...')

        from trainers import const_trainer_train
        _, model = const_trainer_train(config, seed, args)
        model.export_models(args.savepath)

    else:
        raise Exception('Please specify running mode.')
# import os
# import random
# import argparse
# import torch  # Import torch to check for GPU availability

# from pytorch_lightning import seed_everything
# from utils import load_json

# def setup_device(config):
#     if torch.cuda.is_available():
#         os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices
#         print("GPU is available, using GPU.")
#     else:
#         os.environ['CUDA_VISIBLE_DEVICES'] = ''  # No GPU available, use CPU
#         print("GPU not available, using CPU.")

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
#     parser.add_argument('-c', '--config', help='training config file', required=True)
#     parser.add_argument('-t', '--train', action='store_true')
#     parser.add_argument('-i', '--inference', action='store_true')
#     parser.add_argument('-e', '--export', action='store_true')
#     parser.add_argument('-p', '--savepath', default='./')
#     args = parser.parse_args()

#     config = load_json(args.config)
#     setup_device(config)  # Set up the device based on availability

#     seed = config.run.seed or random.randint(1, 10000)
#     seed_everything(seed)
#     print('Using manual seed: {}'.format(seed))
#     print('Config: ', config)

#     if args.train:
#         print('Starting model training...')
#         from trainers import const_trainer_train
#         trainer, model = const_trainer_train(config, seed, args)
#         trainer.fit(model)

#     elif args.inference:
#         print('Starting model inference...')
#         from trainers import const_trainer_test
#         trainer, model = const_trainer_test(config, seed, args)
#         trainer.test(model)

#     elif args.export:
#         print('Exporting saved models...')
#         from trainers import const_trainer_train
#         _, model = const_trainer_train(config, seed, args)
#         model.export_models(args.savepath)

#     else:
#         raise Exception('Please specify running mode.')
