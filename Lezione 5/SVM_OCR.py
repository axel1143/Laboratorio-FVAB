import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Define parameters

SZ = 20  # image size 20x20
bin_n = 16  # Numbers of bins
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR


# Istogramma gradienti orientati = Vettori caratteristiche


# Con le knn il vettore delle caratteristiche era rappresentato direttamente dalla intensità dei pixel.
# Questa volta useremo l'istogramma dei gradienti orientati (HOG) come vettori di caratteristiche.
# Prima di definire la funzione di estrazione degli HOG, raddrizziamo l'immagine usando i suoi momenti del secondo ordine.
# Definiamo prima la funzione deskew() che prende un'immagine di una cifra e la raddrizza.

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


# Applichiamo la tecnica dell'istogramma dei gradienti orientati (HOG) per l'estrazione del vetore delle caratteristiche di ogni immagine.
# Bisogna trovare il descrittore HOG di ogni cella.
# Per questo, applichiamo l'operatore Sobel a ciascuna cella nella direzione X e Y.
# Quindi trova la magnitudine e la direzione del gradiente in ogni pixel.
# Questo gradiente è quantizzato a 16 valori interi.
# Dividiamo l'immagine in quattro sottoquadrati.
# Per ogni sottoquadrato, calcola l'istogramma della direzione (16 bin) ponderato con la loro magnitudine. Quindi ogni sottoquadrato dà un vettore contenente 16 valori.
# Quattro di questi vettori (di quattro sottoquadrati) insieme ci danno un vettore di caratteristiche contenente 64 valori.
# Questo è il vettore delle caratteristiche che utilizziamo per addestrare i nostri dati.

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


# read image od the digits
img = cv.imread('../materiale/digits.png', 0)
if img is None:
    raise Exception("We need digits.png image for samples/data here")

# divide digits image in cells
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# first half is trainData, remaining testData
train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

# apply deskewed and HOG feature extraction functions to training set
deskewed = [list(map(deskew, row)) for row in train_cells]
hogdata = [list(map(hog, row)) for row in deskewed]

trainData = np.float32(hogdata).reshape(-1, 64)

# train data labels
labels = np.repeat(np.arange(10), 250)[:, np.newaxis]

print(trainData.shape)

# create SVM model using svm function from cv.ml.SVM by Opencv library
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)

# set C value
svm.setC(2.67)

# train the model
svm.train(trainData, cv.ml.ROW_SAMPLE, labels)

# save the model
# svm.save('svm_data.dat')

# apply deskewed and HOG feature extraction functions to the test set
deskewed = [list(map(deskew, row)) for row in test_cells]
hogdata = [list(map(hog, row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1, bin_n * 4)

# apply prediction
result = svm.predict(testData)[1]
mask = result == labels

# calculate the accuracy using the match between predictions (result) and true values (labels)
correct = np.count_nonzero(mask)
print(correct*100.0 / result.size)

