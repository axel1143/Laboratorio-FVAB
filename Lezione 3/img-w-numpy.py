import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("../materiale/Img.png", cv2.IMREAD_COLOR)
img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

### Modifica e sotituzione di parti dell'immagine ###

print(img[200, 200])  # accede al singolo pixel (matrice se bgr o unchanged, intensitá se scala di grigi
print(img[200, 200, 0]) # accedo direttamente al canale desiderato (0, 1, 2, 3)

image = plt.figure()

img4[100, 100] = (255, 255, 255) # posso cambiare i pixel
img4[150:200, :, :,]= [0,0,0] # modifico un'intera striscia orizzontale
image.add_subplot(1, 5, 1)
plt.imshow(img4)

img2[150:200, 170:220, :,]= [255,255,0] # modifico un quadrato centrale
image.add_subplot(1, 5, 2)
plt.imshow(img2)

img3[50:100, 180:250, :] = img3[150:200, 180:250, :,] # posso anche replicare alcune parti di un immagine, con la stessa tecnica
img3[150:200, 180:250, :]= [255,255,0]
image.add_subplot(1, 5, 3)
plt.imshow(img3)


print(img2.item(10,10,2)) #utilizziamo funzione item, che peró stampa solo un canale alla volta

### mettere i bordi

replicate = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REPLICATE) ## varie tipologie di bordi possibili
image.add_subplot(1, 5, 4)
plt.imshow(replicate)


replicate = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 255, 0]) ## per border constant anche colore
image.add_subplot(1, 5, 5)
plt.imshow(replicate)
#plt.show()

### resize, traslazione e rotazione immagine

resized = cv2.resize(img,None, fx= 0.5, fy = 0.5, interpolation= cv2.INTER_CUBIC )
# cv2.resize(src, dsize, fattore_asse_x, fattore_asse_y, interpolazione), cosí abbiamo raddoppiato la scala
# se vogliamo scalare in proporzione, non dobbiamo specificare la dsize, basta raddoppiare le due scale
# cv2.imshow("resized", resized), cv2.waitKey(3000)

rows, cols, can = img.shape
M = np.float32([[1,0,100],[0,1,50]]) # necessaria una matrice di traslazione dell'immagine (in questo caso x = 100, y = 50), 1,0 e 0,1 matrice identitá, mentre 100 e 50 vettore traslazione
translated = cv2.warpAffine(img, M, (cols, rows))
#cv2.imshow("traslata", translated), cv2.waitKey(3000)

M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1) #(centro, angolo_rotazione, scala)
rotated = cv2.warpAffine(img, M_rot, (cols, rows))
cv2.imshow("rotated", rotated), cv2.waitKey(3000)

### funzioni di disegno

#cv2.line()
#cv2.circle()
#cv2.rectangle()
#cv2.ellipse()
#cv2.putText()

#argomenti comuni (src, color, thickness, lineType), piú info nella documentazione