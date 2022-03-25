# Caricamento del dataset. In questo semplice esempio utilizzeremo il set di dati sull'inizio del diabete in alcuni
# individui in India. Questo è un set di dati di machine learning standard dal repository UCI Machine Learning.
# Riporta i dati delle cartelle cliniche dei pazienti per gli indiani e se hanno avuto un'insorgenza di diabete entro
# cinque anni.
#
# Si tratta di un problema di classificazione binaria (inizio del diabete 1 o non diabete 0).
# Tutte le variabili di input che descrivono ciascun paziente sono numeriche.
#
# Carichiamo il file (pima-indians-diabetes.csv) come matrice di numeri usando la funzione NumPy loadtxt().
#
# Abbiamo otto variabili di input (otto caratteristiche) e una variabile di output (l'etichetta diabete-non diabete).
# Implementiamo un modello per che mappa le righe di variabili di input (X) sulla base della variabile di output (y)
# si può riassumere come y = f(X).
#
# Le variabili di input (x) sono le seguenti:
#
# 1.   Numero di volte incinta
# 2.   Concentrazione di glucosio plasmatico a 2 ore in un test di tolleranza al glucosio orale
#
# 3. Pressione diastolica (mm Hg)
# 4. Spessore della piega cutanea del tricipite (mm)
# 5. Insulina sierica di 2 ore (mu U/ml)
# 6. Indice di massa corporea (peso in kg/(altezza in m)^2)
# 7. Funzione genealogica del diabete
# 8. Età (anni)
# 9. Variabili di uscita (y):
#
#
# Una volta che il file CSV è stato caricato in memoria, possiamo dividere le colonne di dati in variabili di input e
# output.
#
# I dati verranno archiviati in una matrice 2D in cui la prima dimensione è righe e la seconda dimensione è colonne
import keras.utils.vis_utils
from numpy import loadtxt
from tensorflow import keras as ks
from keras.models import Sequential
from keras import layers

dataset = loadtxt("../materiale/pima-indians-diabetes.csv", delimiter=',')
x_input = dataset[0:537, 0:8] # all columns except for last one, that are labels for each row
y_output = dataset[:537, 8] # last column

print(x_input.shape)
print(y_output.shape)

x_testset = dataset[537:768, 0:8]
y_testset = dataset[537:, 8]

print(x_testset.shape)
print(y_testset.shape)

model = Sequential()
model.add(layers.Dense(12, input_dim=8, activation='relu')) # neural layer with dimension = 12, input dim = 8, activate function = relu
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation= 'sigmoid')) # acti. function = sigmoid
model.summary()

# keras.utils.vis_utils.plot_model(model, "first model with shape", show_shapes=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_input, y_output, epochs=200, batch_size=20,  verbose=1)

loss, accuracy = model.evaluate(x_testset, y_testset)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: %.2f' % (loss*100))

# model.save("path_to_file.h5")
# model = keras.models.load_model("path_to_file.h5")


# make prevision on testset (never seen)
predictions = model.predict(x_testset)
print(predictions[:5])

rounded = [round(x[0]) for x in predictions]
print(rounded[:5])

# compare prediction with expected
for i in range(5):
    print("%s => %d (expected %d)" % (x_testset[i].tolist(), rounded[i], y_testset[i]))