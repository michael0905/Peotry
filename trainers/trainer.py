from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
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
        num_iter_per_epoch = int(len(self.data.poetry_vectors) / self.config.batch_size)
        loop = tqdm(range(num_iter_per_epoch))
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
        print("epoch {}, loss: {}".format(cur_epoch + 1, loss))


    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x_placeholder: batch_x,
                     self.model.y_placeholder: batch_y,
                     self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_op, self.model.loss],
                                     feed_dict=feed_dict)
        return loss
