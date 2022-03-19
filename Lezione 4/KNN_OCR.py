import numpy as np
import cv2 as cv

# -- import image, set to gray scale
img = cv.imread('../materiale/digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# -- now split image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# -- make it into a numpy array with size (50, 100, 20, 20)
x = np.array(cells)

# -- prepare training and test set
train = x[:, :50].reshape(-1, 400).astype(np.float32) # Size = (2500, 400)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

# -- create labels for train and test set
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# -- initiate KNN, train it on training set, then test with k = 5
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbour, dist = knn.findNearest(test, k = 5)

# -- check accuracy classification
# -- compare results with test_labels
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(accuracy)


# -- how to save classification and reload
"""
# Save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)
# Now load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
"""

