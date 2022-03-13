import cv2

#Legge immaigne con parametro
img1 = cv2.imread("../materiale/Img.png", cv2.IMREAD_UNCHANGED)
img2 = cv2.imread("../materiale/Img.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("../materiale/Img.png", cv2.IMREAD_COLOR)

#Mostra immagine in finenstra (lo show, elimina l'alpha)a
cv2.imshow("I1", img1)
cv2.imshow("I2", img2)
cv2.imshow("I3", img3)

#Aspetta venga premuto un tasto e memorizza
key = cv2.waitKey(0)

if key == 27:
    cv2.imwrite("Img_gray.png", img2) #Scrive un immagine, con titolo
    cv2.destroyAllWindows()
elif key == ord('s'):
    cv2.imwrite("Img:unchanged.png",img1)
    cv2.destroyAllWindows()
else:
    cv2.imwrite("Img_color.png", img3)
    # Chiude finestra selezionata
    cv2.destroyWindow("I1")
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
