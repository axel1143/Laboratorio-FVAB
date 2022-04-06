import cv2, numpy as np, matplotlib.pyplot as plt
from skimage.util import random_noise

img = cv2.imread('../materiale/Lena.png')

noise_img = random_noise(img, mode='s&p', amount=0.3) # rumore sale e pepe, fattore 0.3
# https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise rumori possibili

noise_img =  noise_img.astype('float32') # riconvertita per filtri

img_median = cv2.medianBlur(noise_img, 5) # filtro mediana, kernel 5
img_blur = cv2.blur(noise_img, (5, 5)) # filtro media kernel 5x5
img_gauss = cv2.GaussianBlur(noise_img, (5, 5), 0) # filtro gaussiano kernel 5x5
img_bil = cv2.bilateralFilter(noise_img, 5, 75, 75) # filtro bilaterale kernel 5
#mostra
plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)), plt.title('Noise')
plt.subplot(2, 3, 3), plt.imshow(cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB)), plt.title('Median')
plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)), plt.title('Average')
plt.subplot(2, 3, 5), plt.imshow(cv2.cvtColor(img_gauss, cv2.COLOR_BGR2RGB)), plt.title('Gaussian')
plt.subplot(2, 3, 6), plt.imshow(cv2.cvtColor(img_bil, cv2.COLOR_BGR2RGB)), plt.title('Bilateral')

plt.show()