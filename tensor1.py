import tensorflow as tf
import numpy as np

# # This will be an int32 tensor by default; see "dtypes" below.
# rank_0_tensor = tf.constant(4)
#
# # Let's make this a float tensor.
# rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
#
# # If you want to be specific, you can set the dtype (see below) at creation time
# rank_2_tensor = tf.constant([[1, 2],
#                              [3, 4],
#                              [5, 6]], dtype=tf.float16)
#
# # You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:
# #
# #
# np.array(rank_2_tensor)
# rank_2_tensor.numpy()
#
# a = tf.constant([[1, 2],
#                  [3, 4]])
# b = tf.constant([[1, 1],
#                  [1, 1]]) # Could have also said `tf.ones([2,2], dtype=tf.int32)`

# print(tf.add(a, b), "\n")
# print(tf.multiply(a, b), "\n")
# print(tf.matmul(a, b), "\n")

# print(a + b, "\n") # element-wise addition
# print(a * b, "\n") # element-wise multiplication
# print(a @ b, "\n") # matrix multiplication
#
# c = tf.constant([[4.0, 5.0], [10.0, 1.0]])


# # Find the largest value
# print(tf.reduce_max(c))
# # Find the index of the largest value
# print(tf.math.argmax(c))
# # Compute the softmax
# print(tf.nn.softmax(c))

# c = tf.convert_to_tensor([1,2,3])
# print(c)

# d = tf.reduce_max([1,2,3])
# print(d)

# e = tf.reduce_max(np.array([1,2,3]))
# print(e)

rank_4_tensor = tf.zeros([3, 2, 4, 5])