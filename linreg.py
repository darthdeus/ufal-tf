import tensorflow as tf

w = tf.get_variable("w", shape=[2], initializer=tf.initializers.ones())
z = tf.get_variable("z", shape=[2], initializer=tf.initializers.ones())

# x = tf.placeholder(tf.float32, shape=[2], name="x")
# z = x + 1
# zz = x * 13
#
# pi = tf.constant(3.14159, name="pi")
#
# ww = tf.reshape(w, [2, 1])
# xx = tf.reshape(x, [2, 1])
#
# y = tf.matmul(tf.transpose(ww), xx)

init = tf.global_variables_initializer()


# saver = tf.train.Saver()

with open("models/graph.pb", "wb") as f:
    graph_def = tf.get_default_graph().as_graph_def()

    f.write(graph_def.SerializeToString())
    tf.io.write_graph(graph_def, "models", "graph.pbtxt", as_text=True)

    print(graph_def)

# with tf.Session() as sess:
#     sess.run(init)
    # y_ = sess.run(y, {x: [1.0, 1.0]})
    # print(y_)



    # tf.saved_model.simple_save(sess, "models",
    #         inputs={"x": x}, outputs={"y": y})
    # saver.save(sess, "models/model.ckpt")

    # tf.io.write_graph(sess.graph_def, "models", "graph.pb", as_text=False)
    # tf.io.write_graph(tf.get_default_graph(), "models", "graph2.pbtxt")

