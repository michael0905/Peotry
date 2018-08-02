from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.utils import pad_sequence, sample
import tensorflow as tf
from models.generator_model import GeneratorModel


class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        summaries_dict = {
            'train_loss': loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        print("epoch {}, loss: {}".format(cur_epoch, loss))


    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        batch_x, batch_y, batch_length = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x_placeholder: batch_x,
                     self.model.y_placeholder: batch_y,
                     self.model.length_placeholder: batch_length,
                     # self.model.state_placeholder: batch_state,
                     self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_op, self.model.loss],
                                     feed_dict=feed_dict)
        return loss

    def create_poetry(self):
        count = 0
        poetry = ['[']
        x = [self.data.dictionary['[']]
        state = self.sess.run(self.model.stackCell.zero_state(self.config.batch_size, tf.float32))
        while len(poetry) < self.config.longest_length and poetry[-1] not in [']', ' ']:
            feed_x = np.array(pad_sequence(x, self.config.longest_length, self.data.dictionary[' ']))
            feed_x = np.reshape(feed_x, [-1, self.config.longest_length])
            y = np.copy(feed_x)
            length = [1]
            feed_dict = {self.model.x_placeholder: feed_x,
                         self.model.y_placeholder: y,
                         self.model.length_placeholder: length,
                         self.model.state: state,
                         self.model.is_training: False}


            probs, state = self.sess.run([self.model.probs, self.model.state], feed_dict=feed_dict) # [max_time, word_num]

            index = sample(probs[count+1])
            poetry.append(self.data.reversed_dictionary[index])
            x.append(index)
            count += 1
        print(probs[count])
        print(''.join(poetry))
