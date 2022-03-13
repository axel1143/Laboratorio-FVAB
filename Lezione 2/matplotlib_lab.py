import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../materiale/Img.png", cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # usato per invertire i modelli dell'immagine

"""
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]),# per nascondere i valori x ed y degli assi
plt.xlabel("Asse x"), plt.ylabel("Asse y") #aggiungo delle label ad asse x e y
plt.show()
"""

# visualizziamo due immagini affiancate (griglia 1x2)
f = plt.figure()
f.add_subplot(1, 2, 1) # la griglia in cui inseriamo le due immagini (riga, colonna, posizione_corrente)
plt.imshow(img)
plt.xticks([]), plt.yticks([])

f.add_subplot(1, 2, 2) #stessa griglia, posizione 2 e non 1
plt.imshow(img, cmap="gray") #se non inserisco la cmap, l'immagine in scala di grigio viene sfalsata
plt.xticks([]), plt.yticks([])
plt.show()

# cv2 legge immagine secondo il modello (b, g, r) mentre matplotlib legge con (r, g, b), di conseguenza l'immagine
# importata con cv2, può presentare variazioni di colori se mostrata con matplotlib,   caso viene usata la funzione cvtColor
# l'informazione sull'alpha viene mantenuta però in matplotlib
# se però l'immagine è in GRAY_SCALE e cambio in BGR, non ottengo immagine a colore.
