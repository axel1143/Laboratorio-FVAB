import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../materiale/Img.png", cv2.IMREAD_UNCHANGED)
print("Unchanged ",  img.shape) # Restituisce tupla (righe, colonne, canali) se a colori, alpha viene conservato

img3 = cv2.imread("../materiale/Img.png", cv2.IMREAD_COLOR)
print("Colori ", img3.shape) # Se a colori l'alpha si perde

img2 = cv2.imread("../materiale/Img.png", cv2.IMREAD_GRAYSCALE)
print("Scala di grigi ",  img2.shape) # Se in scala di grigi solo (righe, canali)

print(img.size) # Numero totale di elementi (righe x colonne x canali)

print(img.dtype) # Tipi di dati immagine

### Separare e unire i canali

b, g, r, a = cv2.split(img)  #splitta i canali (3 se colori, 4 se unchanged)
img_merged = cv2.merge((b, g, r)) #unisco canali immagine unchanged ma senza, alpha
img_rgb = cv2.merge((r, g, b)) # unisco canali in rgb e non bgr, come unchanged

f = plt.figure()
f.add_subplot(1, 3, 1)
plt.imshow(img)
plt.xticks([]), plt.yticks([])

f.add_subplot(1, 3, 2)
plt.imshow(img_merged)
plt.xticks([]), plt.yticks([])

f.add_subplot(1, 3, 3)
plt.imshow(img_rgb)
plt.xticks([]), plt.yticks([])
plt.show()
