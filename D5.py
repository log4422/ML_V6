#Machine Learning Versuch 6
#Aufgabe D5
#Autor: Armin Sehr
#Bearbeitung: Lukas Götz
#Datum: 16.10.2018


# Dieser Code basiert auf
# https://github.com/ageron/handson-ml/blob/master/10_introduction_to_artificial_neural_networks.ipynb

# Importieren von Bibliotheken
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

# ******** Konstruktion des Berechnungsgraphen *************
# Konstanten, welche die Struktur des Netzwerks festlegen	
n_inputs = 28*28  #Anzahl der Eingangsvariablen = Anzahl der Pixel pro Bild
n_hidden1 = 300   #Anzahl der Einheiten in der ersten verdeckten Schicht
n_hidden2 = 200   #Anzahl der Einheiten in der zweiten verdeckten Schicht
n_hidden3 = 100   #Anzahl der Einheiten in der dritten verdeckten Schicht
n_outputs = 10    #Anzahl der Einheiten in der Ausgabeschicht = Anzahl der Klassen

# Platzhalter für die Eingangsdaten X und die Labels y
# Anzahl der Trainings-Datensaetze bleibt hier offen (shape=(None))
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Laden der Trainings- und Testdaten
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Anzahl der Trainings-/Test-Datensaetze bleibt hier offen (-1)
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0  
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Aufteilen der Trainingsdaten in Trainings- und Validierungsdaten
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Berechnungsgraph fuer das neuronale Netzwerk
# mit tf.layers.dense wird jeweils eine Schicht des Netzwerks aufgebaut
# hidden1 hat X als Eingang, hidden2 hat hidden1 als Eingang usw.
# hidden1 und hidden2 haben ReLu als Aktivierungsfunktion
# logits ist die Ausgabeschicht. Hier wird keine Aktivierungsfunktion ausgewaehlt
# Deshalb muss nachtraeglich die softmax Operation angewendet werden
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden3, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)

# Berechnungsgraph fuer die Kostenfunktion
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Berechnungsgraph fuer die Optimierung der Kostenfunktion (eigentliches Training)
# gewaehlt wird Gradientenabstieg mit Schrittweite 0.01
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Berechnungsgraph zur Ermittlung der Erkennungsrate (accuracy)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# ******** Ausfuehrung des Berechnungsgraphen *************
# Initializsierung der Variablen
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Parameter fuer das stochastische Gradientenverfahren
# In einer Epoche werden alle Trainingsdaten einmal benutzt
# batch_size ist die Anzahl der Daten die fuer eine Iteration benutzt werden
# Wir haben 55000 Trainingsdaten und batch_size = 50:-> 55000/50=1100 Iterationen pro Epoche
# Da wir n_epochs = 40 setzen, benutzen wir alle Trainingsdaten 40mal
n_epochs = 40 
batch_size = 50

# shuffle_batch mischt die Trainingsdaten durch,
# so dass in jeder Epoche die Aufteileung auf die batches
# unterschiedlich ist. Dadurch wird die Gefahr, in einem
# lokalen Minimum stecken zu bleichen reduziert.
# Dadurch verringert sich die Gefahr des Overfittings
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
		
# Durchfuehrung des Trainings
# Aeussere Schleife ueber die Epochen
# Innere Schleife ueber alle Batches innerhalb einer Epoche
# sess.run fuehrt eine Iteration des Gradientenabstiegs durch
# Fuer jede Epoche wird die Erkennungsrate des letzten Batches
# (= Erkennungsrate mit einem Teil der Trainingsdaten)
# und die Erkennungsrate fuer die Validierungsdaten ausgegeben
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "G:\Semester_7\ML_V6/my_model_final.ckpt")
	
# Klassifikation der ersten 20 Testdaten und Vergleich der Ergebnisse mit den Labels	
with tf.Session() as sess:
    saver.restore(sess, "G:\Semester_7\ML_V6/my_model_final.ckpt") # or better, use save_path
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
	
print("Predicted classes:", y_pred)
print("Actual classes:   ", y_test[:20])
