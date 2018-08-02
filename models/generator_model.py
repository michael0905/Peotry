from base.base_model import BaseModel
import tensorflow as tf


class GeneratorModel(BaseModel):
    def __init__(self, config, word_num):
        super(GeneratorModel, self).__init__(config)
        self.word_num = word_num
        self.build_model(config)
        self.init_saver()


    def build_model(self, config):
        self.config = config
        # here you build the tensorflow graph of any model you want and also define the loss.
        self.x_placeholder = tf.placeholder(tf.int32, [self.config.batch_size, self.config.longest_length], name='x')
        self.y_placeholder = tf.placeholder(tf.int32, [self.config.batch_size, self.config.longest_length], name='y')
        self.length_placeholder = tf.placeholder(tf.int32, [None], name='length')
        self.is_training = tf.placeholder(tf.bool, name='training')
        # self.state_placeholder = tf.placeholder(tf.float32, [None, self.config.state_size], name='state')
        self.embedded_text = self.add_embedding()
        self.probs, self.logits, self.state = self.add_prediction_op()
        self.loss = self.add_loss_op(self.logits)
        self.train_op = self.add_training_op(self.loss)

    def add_embedding(self):
        with tf.variable_scope('embedding'):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.word_num, self.config.embedding_size], -1.0, 1.0))
            embedded_text = tf.nn.embedding_lookup(self.embeddings, self.x_placeholder)
            # embedded_text_expanded = tf.expand_dims(embedded_text, axis=-1)
            return embedded_text

    def add_prediction_op(self):
        """
        probability over all dictionary words
        """
        with tf.variable_scope('rnn'):
            # cell_fw = LSTMCell(self.config.state_size)
            # cell_bw = LSTMCell(self.config.state_size)
            # # rnn_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
            # outputs, (c_state, m_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedded_text,
            #                     sequence_length=self.x_length_placeholder, dtype=tf.float32)
            # output = tf.concat(outputs, axis=2)
            basicCell = tf.contrib.rnn.BasicLSTMCell(self.config.state_size, state_is_tuple = False)
            # self.stackCell = tf.contrib.rnn.MultiRNNCell([basicCell] * 2)
            self.stackCell = basicCell
            self.state = self.stackCell.zero_state(self.config.batch_size, tf.float32)
            outputs, finalState = tf.nn.dynamic_rnn(self.stackCell, self.embedded_text, \
                sequence_length=self.length_placeholder, initial_state=self.state, dtype=tf.float32)
            outputs = tf.reshape(outputs, [-1, self.config.state_size]) # [batch_size*max_time, state_size]
        with tf.variable_scope("softmax"):
            w = tf.get_variable("w", [self.config.state_size, self.word_num])
            b = tf.get_variable("b", [self.word_num])
            logits = tf.matmul(outputs, w) + b
            probs = tf.nn.softmax(logits)
            print(probs.shape)
        return probs, logits, finalState # [batch_size*max_time, word_num]

    def add_loss_op(self, logits):
        with tf.variable_scope('loss'):
            targets = tf.reshape(self.y_placeholder, [-1]) #[batch_size*max_time]
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                                                                  [tf.ones_like(targets, dtype=tf.float32)])
            loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        with tf.variable_scope('train'):
            trainableVariables = tf.trainable_variables()
            grads, a = tf.clip_by_global_norm(tf.gradients(loss, trainableVariables), 5)

            learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step=self.global_step_tensor,
                                                  decay_steps=self.config.decay_steps, decay_rate=self.config.decay_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, trainableVariables), global_step=self.global_step_tensor)
        return train_op

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
