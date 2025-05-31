import tensorflow as tf
import numpy as np

# # This will be an int32 tensor by default; see "dtypes" below.
# rank_0_tensor = tf.constant(4)
#
# # Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
#
# # If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
#
# # You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:
# #
# #
# np.array(rank_2_tensor)
# rank_2_tensor.numpy()

rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

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

# rank_4_tensor = tf.zeros([3, 2, 4, 5])
# print("Type of every element:", rank_4_tensor.dtype)
# print("Number of axes:", rank_4_tensor.ndim)
# print("Shape of tensor:", rank_4_tensor.shape)
# print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
# print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
# print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# # Indexing
# rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
# print(rank_1_tensor.numpy())
# print("First:", rank_1_tensor[0].numpy())
# print("Second:", rank_1_tensor[1].numpy())
# print("Last:", rank_1_tensor[-1].numpy())
# # Indexing with a : slice keeps the axis:
# print("Everything:", rank_1_tensor[:].numpy())
# print("Before 4:", rank_1_tensor[:4].numpy())
# print("From 4 to the end:", rank_1_tensor[4:].numpy())
# print("From 2, before 7:", rank_1_tensor[2:7].numpy())
# print("Every other item:", rank_1_tensor[::2].numpy())
# print("Reversed:", rank_1_tensor[::-1].numpy())

# Multi-axis indexing
# print(rank_2_tensor[1, 1].numpy())
# print("Second row:", rank_2_tensor[1, :].numpy())
# print("Second column:", rank_2_tensor[:, 1].numpy())
# print("Last row:", rank_2_tensor[-1, :].numpy())
# print("First item in last column:", rank_2_tensor[0, -1].numpy())
# print("Skip the first row:")
# print(rank_2_tensor[1:, :].numpy(), "\n")
#
# print(rank_3_tensor[:, :, 4])

# Manipulating Shapes
# Shape returns a `TensorShape` object that shows the size along each axis
# x = tf.constant([[1], [2], [3]])
# # print(x.shape)
# # # You can convert this object into a Python list, too
# # print(x.shape.as_list())
# # You can reshape a tensor to a new shape.
# # Note that you're passing in a list
# reshaped = tf.reshape(x, [1, 3])
# # print(x.shape)
# # print(reshaped.shape)
#
# # A `-1` passed in the `shape` argument says "Whatever fits".
# print(tf.reshape(rank_3_tensor, [-1]))

# For this 3x2x5 tensor,
# reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do,
# as the slices do not mix:
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
