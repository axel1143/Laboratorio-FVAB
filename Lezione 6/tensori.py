import tensorflow as tf
import numpy as np

# rank 0 tensor
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# rank 1 tensor
rank_1_tensor = tf.constant([1.0, 2.0, 3.0])
print(rank_1_tensor)

# rank 2 tensor / matrice
rank_2_tensor = tf.constant([[1, 2],
                            [3, 4],
                            [5, 6]], dtype=tf.float16)  # cambio del datatype
print(rank_2_tensor)

# rank 3/n tensor (tensore 3 x 2 x 3)
rank_3_tensor = tf.constant([[[1, 2, 3],
                              [4, 5, 6]],
                             [[7, 8, 9],
                              [10, 11, 12]],
                             [[13, 14, 15],
                              [16, 17, 18]], ])
print(rank_3_tensor)

# tensors from numpy
a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a, t_b)

