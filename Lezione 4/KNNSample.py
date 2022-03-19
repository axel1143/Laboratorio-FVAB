import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -- defining training set
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)  # 25 cordinate (x, y) 2 dim, da 0 a 100

# -- defining label for training set
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)  # 25 etichette 0/1, 1 dim


# -- extract red (label = 0)
red = trainData[responses.ravel() == 0]
## plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

# -- extract blue (label = 1)
blue = trainData[responses.ravel() == 1]
## plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

# -- generate newcomer to classifier
newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
## plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o' )


# --  Starting with KNN, with our training Data, and neighbour = 3
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

# -- Print results
print("result:  {}\n".format(results))
print("neighbours:  {}\n".format(neighbours))
print("distance:  {}\n".format(dist))
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')
plt.show()
