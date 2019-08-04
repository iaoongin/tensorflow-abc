# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf

# var
state = tf.Variable(0, name="num")

one = tf.constant(1)

update = tf.assign_add(state, one)

init_var = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_var)

    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
