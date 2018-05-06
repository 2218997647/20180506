import tensorflow as tf

import numpy as np
import gc
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data



data = input_data.read_data_sets('data/fashion')


#'c'的outputmaps是convolution之后有多少张图，比如上(最上那张经典的))第一层convolution之后就有六个特征图,'c'的kernelsize 其实就是用来convolution的patch是多大,'s'的scale就是pooling的size为scale*scale的区域

image_size1 = 28
image_size2 = 28
num_labels = 10
num_channels = 1

batch_size = 128
patch_size = 3
depth = 72
num_hidden = 128
learning_rate=0.001


train=data.train
train_dataset,train_labels=train.images,train.labels
valid_dataset,valid_labels=data.validation.images,data.validation.labels
test_dataset,test_labels=data.test.images,data.test.labels


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size1, image_size2, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset,train_labels=reformat(train.images,train.labels)
valid_dataset,valid_labels=reformat(data.validation.images,data.validation.labels)
test_dataset,test_labels=reformat(data.test.images,data.test.labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
graph = tf.Graph()


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


with graph.as_default():
    # Input data.


    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size1, image_size2, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    keep_prob = tf.placeholder(tf.float32)


    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size1 // 4 * image_size2 // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))



    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv=tf.nn.dropout(conv,keep_prob=0.35)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.dropout(conv, keep_prob=0.35)
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases



    logits = model(tf_train_dataset)
    #test_logits = model(tf_test_dataset)





    ##print(logits)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        tf.scalar_summary('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               100, 0.95)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 16800
with tf.Session(graph=graph) as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/", session.graph)
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob: 0.5}
        result, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            writer.add_summary(result, step)
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

