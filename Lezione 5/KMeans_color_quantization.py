# La quantizzazione del colore è il processo di riduzione del numero di colori in un'immagine. Usiamo l'algoritmo di
# clustering k-means per la quantizzazione del colore.

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
Funzione di qunantizzazione:
1) Ogni pixel = vettore tridimensionale RGB. Prima fatto un reshape matrice colori ad array taglia MX3 dove M è il numero di pixel nell'immagine
2) Settiamo poi i criteri dell'algoritmo: 20 iterazioni = stop
3) cv.TERM_CRITERIA_EPS - interrompe l'iterazione dell'algoritmo, se raggiunta precisione specificata, epsilon = 1.0
4) Applichiamo k-means e salviamo in center i centroidi del cluster trovati
5) Convertiamo in unit8 e ricostuiamo l'immagine originale 
"""


def quant(img, k):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result


img = cv.imread("../materiale/Lena.png")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img)
axs[0, 0].set_title("original")
axs[0, 1].imshow(quant(img, 2))
axs[0, 1].set_title("k=2")
axs[1, 0].imshow(quant(img, 4))
axs[1, 0].set_title("k=4")
axs[1, 1].imshow(quant(img, 8))
axs[1, 1].set_title("k=8")

fig.tight_layout()
plt.show()
