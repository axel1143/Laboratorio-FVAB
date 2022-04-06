import cv2, numpy as np, matplotlib.pyplot as plt

img = cv2.imread("../materiale/Lena.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width, col = img.shape

# resize
img_small = cv2.resize(img, (int(width/10), int(height/10)), interpolation=cv2.INTER_CUBIC) # dimensione finale deve essere intera
img_big =  cv2.resize(img, (2*width, 2*height))

# traslazione
M = np.float32([[1, 0, 100],[0, 1, 50]])
print(M)
img_translated = cv2.warpAffine(img, M, (width, height))

# rotazione
M = cv2.getRotationMatrix2D((int(width/2), int(height/2)), 180, 1)
print(M)
img_rot = cv2.warpAffine(img, M, (width, height))

# mostriamo

plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Originale')
plt.subplot(2, 3, 2), plt.imshow(gray, cmap='gray'), plt.title('Gray') #uso cmap per mostrare correttamente i grigi
plt.subplot(2, 3, 3), plt.imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)), plt.title('Small')
plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(img_big, cv2.COLOR_BGR2RGB)), plt.title('Big')
plt.subplot(2, 3, 5), plt.imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB)), plt.title('Traslation')
plt.subplot(2, 3, 6), plt.imshow(cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)), plt.title('Rotation')

plt.show()