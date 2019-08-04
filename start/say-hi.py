import os

import tensorflow as tf

# 忽略警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
say = tf.constant("jsut say hi !")

with tf.compat.v1.Session() as sess:

    run = sess.run(say)
    print(run)
