import tensorflow as tf
import tqdm
import os
import logging
from collections import namedtuple


class SkipGrams:
    def __init__(self, config, inputs):
        self.initializer = inputs.initializer
        self.contexts = inputs.contexts
        self.targets = inputs.targets

        self.config = config

        self._build_graph()

    def _embed(self):
        self.embeddings = tf.get_variable('embeddings',
                                          shape=[self.config.vocab_size, self.config.embed_size],
                                          dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(-1.0, 1.0))
        self.context_embed = tf.nn.embedding_lookup(self.embeddings, self.contexts)

    def _compute_loss(self):
        weights = tf.get_variable('weights',
                                  shape=[self.config.vocab_size, self.config.embed_size],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases',
                                 shape=[self.config.vocab_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer())

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights, biases, tf.expand_dims(self.targets, axis=-1), self.context_embed,
                                       self.config.num_sampled, self.config.vocab_size)
        )

    def _add_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, self.global_step)

    def _add_summaries(self):
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._embed()
        self._compute_loss()
        self._add_train_op()
        self._add_summaries()

    def train_epoch(self, sess, epoch, writer, summary_freq):
        sess.run(self.initializer)
        step = 0
        t = tqdm.trange(self.config.data_size, desc='Epoch {}'.format(epoch))
        while True:
            try:
                _, loss, summaries, global_step = sess.run([self.train_op, self.loss, self.summary_op, self.global_step])
                step += 1

            except tf.errors.OutOfRangeError:
                t.update(self.config.train_size % summary_freq)
                t.close()
                break

            if summary_freq > 0 and step % summary_freq == 0:
                t.update(summary_freq)
                t.set_postfix_str('loss: {:.4f}'.format(loss))
                writer.add_summary(summaries, global_step)

    def train(self):
        saver = tf.train.Saver(max_to_keep=1)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.tables_initializer())

            if self.config.restore_from:
                if os.path.isdir(self.config.restore_from):
                    save_path = tf.train.latest_checkpoint(self.config.restore_from)
                else:
                    save_path = self.config.restore_from
                saver.restore(sess, save_path)

            else:
                sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(self.config.summary_dir, sess.graph)

            for epoch in range(1, self.config.num_epochs + 1):
                self.train_epoch(sess, epoch, writer, self.config.summary_freq)
                save_path = os.path.join(self.config.ckpt_dir, 'after_epoch')
                saver.save(sess, save_path=save_path, global_step=epoch)


if __name__ == '__main__':
    from vocab import Vocab
    from input_fn import build_skip_grams_inputs

    vocab = Vocab(['../data/toy_data.txt'])
    vocab.filter_by_size(10000)

    config = {
        'vocab_size': 10000,
        'embed_size': 50,
        'num_sampled': 10,
        'learning_rate': 0.001,
        'data_size': 10000 * 2 // 8 + 1,
        'num_epochs': 5,
        'num_parallel_calls': 8,
        'batch_size': 8,
        'num_skips': 2,
        'skip_window': 1
    }
    config = namedtuple('Config', config.keys())(**config)
    inputs = build_skip_grams_inputs('../data/toy_data.txt', config, vocab)
    model = SkipGrams(config, inputs)

    model.train()

