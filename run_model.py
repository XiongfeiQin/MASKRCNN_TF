"""
function；training and testing MRCNN
date:
author: Junior Liu
"""

import tensorflow as tf
import Network
from ResNet50 import ResNet50
import numpy as np


path = 'E:/fingerprint/0320/'  # path of .npy and .tfrecord files
np.set_printoptions(threshold=np.inf)
# for training
train_mode = tf.placeholder(tf.bool)
batches = tf.placeholder(tf.float32, [None, 160, 160, 1])
labels = tf.placeholder(tf.float32, [None, 20, 20, 3])
#
vgg = Network.Network(path+'test3-save.npy')
# vgg = fingerprit.FingerPrint()#training from beginning
learning_rate = 0.000001

num_epoch = 1
batch_size = 20
capacity = 100000  #num of total samples
min_after_dequeue = 30000
# Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
# img1 = utils.load_image("E:/fingerprint/TPBmp/CSZTP00000064_01_3.bmp")
# img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
# batch1 = img1.reshape((1, 128, 128, 1))


vgg.build(batches, batch_size, train_mode)
# print number of variables used
print("num of variables:", vgg.get_var_count())

# test classification
# prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
# utils.print_prob(prob[0], './synset.txt')

temp = tf.reduce_sum((1 - (20 * labels - 1) ** 2) * 100. * (labels - vgg.fc9) ** 2, reduction_indices=[1, 2, 3])\
+ tf.reduce_sum(tf.nn.relu(vgg.fc9 - 0.1)+tf.nn.relu(-vgg.fc9), reduction_indices=[1, 2, 3])\
       # + tf.reduce_sum(tf.nn.relu(tf.abs(labels-vgg.fc9)-4e-4))   # allow error within 0.1/128

cross_entropy = tf.reduce_mean(temp)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

label = vgg.getoutput(labels)
orientation = vgg.getoutput(vgg.fc9)
correct_prediction = tf.reduce_mean(tf.abs(label-orientation))/179.
accuracy = tf.reduce_mean(1 - correct_prediction)

# reading data
image_flow, label_flow = vgg.read_and_decode(path+'train_data3.tfrecords', num_epoch)
img_batch, label_batch = tf.train.shuffle_batch\
    ([image_flow, label_flow], batch_size=batch_size,
     capacity=capacity, min_after_dequeue=min_after_dequeue)
#training
# init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
# init_op = tf.global_variables_initializer()
with tf.device('/gpu:0'):
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # sess.run(init_op)
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    try:
        while not coord.should_stop():
            get_batches, get_labels = sess.run([img_batch, label_batch])
            # print(temp.get_shape())
            # print(sess.run(temp, feed_dict={batches: get_batches, labels: get_labels, train_mode: False}))
            # print("batches:", sess.run((tf.reduce_max(batches), tf.reduce_min(batches)),
            #               feed_dict={batches: get_batches}))
            # print("fc9:", sess.run((tf.reduce_max(vgg.fc9), tf.reduce_min(vgg.fc9)),
            #                        feed_dict={batches: get_batches, train_mode: False}))
            # print("labels:", sess.run((tf.reduce_max(labels), tf.reduce_min(labels)), feed_dict={labels: get_labels}))
            # if i == 0:
            #     print("fc9:", sess.run((vgg.getoutput(vgg.fc9)),
            #                            feed_dict={batches: get_batches, labels: get_labels, train_mode: True}))
            #     print("labels:", sess.run(vgg.getoutput(labels), feed_dict={labels: get_labels}))

            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={batches: get_batches, labels: get_labels, train_mode: False})
                print("step %d, accuracy %g" % (i, train_accuracy))
            loss, _ = sess.run([cross_entropy, train_step],
                                feed_dict={batches: get_batches, labels: get_labels, train_mode: True})
            if i % 10 == 0:
                print("%dth batch: loss=%f" % (i, loss))
            if i % 1000 == 0:
                vgg.save_npy(sess, path + 'test3-save.npy')
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        # test save
        vgg.save_npy(sess, path+'test3-save.npy')
        coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()

