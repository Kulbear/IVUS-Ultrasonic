from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class SigmoidTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SigmoidTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        ep = self.sess.run(self.model.cur_epoch_tensor)
        print('Epoch:', ep)
        loop = tqdm(range(self.config.num_iter_per_epoch))

        # train and evaluate on the training set
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)
        print('Train Loss: {}'.format(loss))

        # evaluate on the test set
        test_loss = self.test()
        print('Test Loss: {}'.format(test_loss))

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}

        if test_loss == 1.0:
            raise ZeroDivisionError

        # only save the best model
        if test_loss < self.best_test_result[1]:
            self.best_test_result = [
                ep, test_loss, {
                    'loss': loss,
                    'test_loss': test_loss,
                }
            ]
            self.model.save(self.sess)
            print(self.best_test_result)

        summaries_dict['training_loss'] = loss
        summaries_dict['test_loss'] = test_loss

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def test(self):
        test_losses = []
        print('Testing...')
        test_subsets = self.data.get_test_subsets(self.config.batch_size)
        for subset in test_subsets:
            test_x, test_y = subset
            feed_dict = {
                self.model.x: test_x,
                self.model.y: test_y,
                self.model.is_training: False
            }
            loss = self.sess.run(self.model.cross_entropy, feed_dict=feed_dict)
            loss = loss * test_x.shape[0]
            test_losses.append(loss)

        return np.sum(test_losses) / self.data.test_input.shape[0]

    def train_step(self):
        batch_x, batch_y = self.data.next_batch(self.config.batch_size)
        feed_dict = {
            self.model.x: np.expand_dims(batch_x, axis=-1),
            self.model.y: batch_y,
            self.model.is_training: True
        }
        _, loss = self.sess.run(
            [self.model.train_step, self.model.cross_entropy],
            feed_dict=feed_dict)

        return loss
