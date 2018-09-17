import numpy as np
import tensorflow as tf


class LongShortTermMemoryModel:
    def __init__(self, encodings_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [])  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encodings_size])  # Shape: [batch_size, max_time, encodings_size]
        self.y = tf.placeholder(tf.float32, [None, None, encodings_size])  # Shape: [batch_size, max_time, encodings_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, encodings_size]))
        b = tf.Variable(tf.random_normal([encodings_size]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.einsum('bts,se->bte', lstm, W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


char_encodings = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ' ' 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'a' 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'b' 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'c' 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'e' 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'h' 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'i' 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'k' 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'n' 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'o' 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 's' 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 't' 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 'u' 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 'y' 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 'sun' 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 'son' 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 'cat' 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 'hat' 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 'cake' 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 'bike' 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 'key' 20
]
index_to_char = [' ', 'a', 'b', 'c', 'e', 'h', 'i', 'k', 'n', 'o', 's', 't', 'u', 'y',
                 '\U00002600', '\U0001F466', '\U0001F408', '\U0001F3A9', '\U0001F370', '\U0001F6B2', '\U0001F511']

x_train = [
    [char_encodings[10], char_encodings[12], char_encodings[8],  char_encodings[0]],  # 'sun ' 0
    [char_encodings[10], char_encodings[9],  char_encodings[8],  char_encodings[0]],  # 'son ' 1
    [char_encodings[3],  char_encodings[1],  char_encodings[11], char_encodings[0]],  # 'cat ' 2
    [char_encodings[5],  char_encodings[1],  char_encodings[11], char_encodings[0]],  # 'hat ' 3
    [char_encodings[3],  char_encodings[1],  char_encodings[7],  char_encodings[4]],  # 'cake' 4
    [char_encodings[2],  char_encodings[6],  char_encodings[7],  char_encodings[4]],  # 'bike' 5
    [char_encodings[7],  char_encodings[4],  char_encodings[13], char_encodings[0]]   # 'key ' 6
]
y_train = [
    [char_encodings[14]],  # 'Sun'
    [char_encodings[15]],  # 'Son'
    [char_encodings[16]],  # 'Cat'
    [char_encodings[17]],  # 'Hat'
    [char_encodings[18]],  # 'Cake'
    [char_encodings[19]],  # 'Bike'
    [char_encodings[20]]   # 'Key'
]

model = LongShortTermMemoryModel(np.shape(char_encodings)[1])

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session object for running TensorFlow operations
with tf.Session() as sess:
    # Initialize tf.Variable objects
    sess.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = sess.run(model.in_state, {model.batch_size: 7})

    for epoch in range(500):
        sess.run(minimize_operation, {model.batch_size: 7, model.x: x_train, model.y: y_train, model.in_state: zero_state})

        if epoch % 10 == 9:
            print("Epoch", epoch)
            print("Loss", sess.run(model.loss, {model.batch_size: 7, model.x: x_train, model.y: y_train, model.in_state: zero_state}))

            # Generate characters from the initial characters ' h'
            state = sess.run(model.in_state, {model.batch_size: 7})
            text = ""
            _, state = sess.run([model.f, model.out_state], {model.batch_size: 7, model.x: [[char_encodings[0]]], model.in_state: state})  # ' '
            y, state = sess.run([model.f, model.out_state], {model.batch_size: 7, model.x: [[char_encodings[10]]], model.in_state: state})  # 's'
            text += index_to_char[y.argmax()]
            # for c in range(50):
            y, state = sess.run([model.f, model.out_state], {model.batch_size: 7, model.x: [[char_encodings[y.argmax()]]], model.in_state: state})
            text += index_to_char[y[0].argmax()]
            print(text)
