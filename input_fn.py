import tensorflow as tf
import tensorflow.contrib as tc
import collections
import random
from functools import partial


BatchedInputs = collections.namedtuple('BatchedInputs',
                                       ['initializer', 'contexts', 'targets'])


def build_skip_grams_inputs(config, vocab):
    table = tc.lookup.index_table_from_tensor(vocab.id2token, default_value=vocab.token2id[vocab.unk_token])

    with open(config.data_path) as f:
        data = f.readline().split()
    generator = partial(gen_batchv2,
                        data=data,
                        num_skips=config.num_skips,
                        skip_window=config.skip_window)
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=tf.string)
    dataset = dataset.map(lambda x: table.lookup(x), num_parallel_calls=config.num_parallel_calls)
    dataset = dataset.map(lambda x: tf.unstack(x, num=2, axis=-1), num_parallel_calls=config.num_parallel_calls)
    dataset = dataset.batch(config.batch_size)
    iterator = dataset.make_initializable_iterator()
    contexts, targets = iterator.get_next()
    return BatchedInputs(iterator.initializer, contexts, targets)


def gen_batch(data, batch_size, num_skips, skip_window):
    """
    Given a corpus, generate batch of data, each element is a pair of (context, target).
    The code below mainly follows the official tensorflow tutorials from:
    https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    data_index = 0
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    max_index = len(data)

    buffer = collections.deque(maxlen=span)

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    while data_index <= max_index:
        batch = []
        for i in range(batch_size // num_skips):
            target_word = buffer[skip_window]
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)      # here just use the simple uniform random sample
            for j, context_word in enumerate(words_to_use):
                batch.append([buffer[context_word], target_word])

            if data_index == max_index:
                data_index += 1
                break
            else:
                buffer.append(data[data_index])
                data_index += 1
        yield batch


def gen_batchv2(data, num_skips, skip_window):
    assert num_skips <= 2 * skip_window

    data_index = 0

    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    buffer.extend(data[data_index: data_index + span])
    data_index += span

    while True:
        target_word = buffer[skip_window]
        possible_indexes = [w for w in range(span) if w != skip_window]
        context_words = [buffer[i] for i in random.sample(possible_indexes, num_skips)]
        for context_word in context_words:
            yield [target_word, context_word]

        if data_index == len(data):
            break
        else:
            buffer.append(data[data_index])
            data_index += 1
