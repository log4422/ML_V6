#Machine Learning Versuch 6
#Aufgabe D6
#Version 1.2
#Autor: Lukas Götz
#Datum: 24.10.2018


#Einbinden aller nötigen Bibliotheken-----------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#-----------------------------------------------------------------------------------------------------------------------



#Laden der Daten--------------------------------------------------------------------------------------------------------
IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int32, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST, target_dtype=np.int32, features_dtype=np.float32)

#Zuweisung der Daten in train und valid
X_train = training_set.data
y_train = training_set.target
X_valid = test_set.data
y_valid = test_set.target

#Erstellen der Platzhalter für X und y
X = tf.placeholder(tf.float32, shape=(None, 4), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
#-----------------------------------------------------------------------------------------------------------------------



#Erstellung des Berechnungsgraph----------------------------------------------------------------------------------------
#Definition von Konstanten
n_hidden1 = 4
n_hidden2 = 4
n_outputs = 3
learning_rate =0.01


#Struktur des DNN
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="output")
    h = tf.nn.softmax(logits)

#Berechnungsgraph der Kostenfunktion
with tf.name_scope("loss"):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

#Berechnungsgraph der Optimierung der Kostenfunktion
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#Berechnungsgraph zur Ermittlung der Erkennungsrate
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuarcy = tf.reduce_mean(tf.cast(correct, tf.float32))
#-----------------------------------------------------------------------------------------------------------------------


#Ausführen des Berechnungsgraphen---------------------------------------------------------------------------------------

#Initialisierung der Variablen
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Festlegung Durchläufe
n_epochs = 80
batch_size = 4

#Durchmischen der Trainingsdaten
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

#Durchführung des Trainings
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X :X_batch, y: y_batch})
        acc_batch = accuarcy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuarcy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "G:\Semester_7\ML_V6/my_model_final.ckpt")

#Klassifikation der ersten 10 Testdaten
with tf.Session() as sess:
    saver.restore(sess, "G:\Semester_7\ML_V6/my_model_final.ckpt")
    X_new_scaled = X_valid[:10]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print("Predicted classes:", y_pred)
print("Actual classes:   ", y_valid[:10])
#-----------------------------------------------------------------------------------------------------------------------
