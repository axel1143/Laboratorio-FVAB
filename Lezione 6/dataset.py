
import tensorflow as tf
from tensorflow import keras as ks

# dataset semplice
dataset0 = tf.data.Dataset.from_tensors([1, 2, 3])
dataset_sl = tf.data.Dataset.from_tensor_slices(
    [1, 2, 3])  # -> Creates a Dataset whose elements are slices of the given tensors.
# The given tensors are sliced along their first dimension.

print("Tensore rank 1 (3 colonne) , diviso in 3 Tensori rank 0")
for element in dataset_sl:
    print(element)

print("Tensore rank 1")
print(dataset0[0])

# dataset con tensori
rank_2_tensors = tf.constant([[1, 2],
                              [3, 4],
                              [5, 6]])

dataset1 = tf.data.Dataset.from_tensors(rank_2_tensors)
print("Tensore rank 2")
print(dataset1[0])

# caricamento dataset da disco
# datset_files = tf.data.TextLineDataset(["file1.txt", "file2.txt"])

# se formato TFRecords
# dataset_records = tf.data.TFRecordDataser(["file1.tfrecords", "file2.tfrecords"])

# caricamento file con keras
directory = "file_path"
# ks.utils.image_dataset_from_directory(
#     directory,
#     labels = "inferred",
#     label_mode = "int",
#     class_names = None,
#     color_mode = 'rgb',
#     batch_size = 32,
#     image_size = (256, 256),
#     shuffle = True,
#     seed = None,
#     validation_split = None,
#     subset = None,
#     interpolation = "bilinear",
#     follow_links = False,
#     crop_to_aspect_ratio = False,
# )
