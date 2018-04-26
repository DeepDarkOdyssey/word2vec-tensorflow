import argparse
import os
import json
from munch import Munch
from vocab import Vocab
from input_fn import build_skip_grams_inputs
from utils import load_config, prepare_experiment
from models import SkipGrams

parser = argparse.ArgumentParser()
parser.add_argument('--prepare', action='store_true',
                    help='Build vocabulary and directories')
parser.add_argument('--train', action='store_true',
                    help='Train the model')

# experiment configuration
parser.add_argument('--experiments_dir', type=str, default='experiments/',
                    help='the directory to store all the experiments data')
parser.add_argument('--model_name', type=str, default='SkipGrams',
                    help='the name of the model')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='the unique name of this experiment')
parser.add_argument('--config_path', type=str, default='',
                    help='path to load config that has been saved before, if specified')
parser.add_argument('--fixed_configs', nargs='+', type=str,
                    default=['config_path'],
                    help='configs need to be fixed when loading another config')

# preparing settings
parser.add_argument('--data_path', type=str, default='data/text8/text8.txt',
                    help='Path to the corpus file you want to build word2vec from.')
parser.add_argument('--data_dir', type=str, default='data/text8',
                    help='the directory to save the the data correlated files')
parser.add_argument('--vocab_size', type=int, default=50000,
                    help='The max amount of word you want to maintain in your vocab.')
parser.add_argument('--vocab_dir', type=str, default='data/text8/vocab',
                    help='the directory to store the built vocabulary')
parser.add_argument('--global_config', type=str, default='global_config.json',
                    help='path to save some configurations after preparation')

# dataset configuration
parser.add_argument('--num_parallel_calls', type=int, default=8,
                    help='How many threads to use while processing the dataset')
parser.add_argument('--batch_size', type=int, default=128,
                    help='How many samples contained in one batch.')
parser.add_argument('--num_skips', type=int, default=2,
                    help='How many times to reuse an input to generate a label.')
parser.add_argument('--skip_window', type=int, default=1,
                    help='How many words to consider left and right.')
parser.add_argument('--num_sampled', type=int, default=64,
                    help='Number of negative examples to sample.')

# model hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='the learning rate of optimization algorithms')
parser.add_argument('--embed_size', type=int, default=128,
                    help='Dimension of the embedding vector.')

# training settings
parser.add_argument('--num_epochs', type=int, default=3,
                    help='number of training epochs')
parser.add_argument('--restore_from', type=str, default='',
                    help="the checkpoint file or the directory it's stored in")
parser.add_argument('--summary_freq', type=int, default=10,
                    help='frequency of saving summaries')


def run():
    args = parser.parse_args()
    config = Munch().fromDict(vars(args))

    print(config)

    if config.prepare:
        # make sure every directory in config exists
        for key, value in vars(config).items():
            if key.endswith('dir') and not os.path.exists(value):
                os.mkdir(value)

        print('Building vocabularies...')
        vocab = Vocab(fins=[config.data_path], sep=' ')
        vocab.filter_by_size(config.vocab_size)
        vocab.save_to(os.path.join(config.vocab_dir, 'vocab.data'))
        print('Saving vocab to {}'.format(config.vocab_dir))

        with open(config.data_path) as f:
            data_size = len(f.readline().split())

        global_config = {
            'data_size': data_size,
            'vocab_size': vocab.size
        }
        with open(config.global_config, 'w') as f:
            json.dump(global_config, f, indent=4)
        print('Saving global config to {}'.format(config.global_config))

    if config.train:
        config = load_config(config, config.global_config)

        config = prepare_experiment(config)

        # save current config to the experiment directory
        with open(os.path.join(config.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # set_logger(config)

        vocab = Vocab()
        vocab.load_from(os.path.join(config.vocab_dir, 'vocab.data'))

        inputs = build_skip_grams_inputs(config, vocab)
        model = SkipGrams(config, inputs)

        model.train()

if __name__ == '__main__':
    run()
