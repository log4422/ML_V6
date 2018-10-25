# -----------------------------------------------------------------------------------------------------------------------
# Autor: Lukas Götz
# Softmax-Regression v2.1
# Datum: 20.10.2018
# -----------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------
# Initialierungen von Tensorflow, Numpy und Matplotlib, Setup
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

# Deaktivierung von Warnungen
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import der MNIST Daten
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# -----------------------------------------------------------------------------------------------------------------------


# Festlegung der Modelparameter------------------------------------------------------------------------------------------
learning_rate = 0.7
batch_size = 128
n_epochs = 500
size_image = 784
n_class = 10

# Definiton von Platzhaltern X und y um das Modell mit Testdaten zu füttern
X = tf.placeholder(dtype=tf.float32, shape=(batch_size, size_image), name="X")
y = tf.placeholder(dtype=tf.int8, shape=(batch_size, n_class), name="y")

# Definition von weights and bias
w = tf.get_variable(name="weights", shape=(size_image, n_class), dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(-5, 5))
b = tf.get_variable(name="bias", shape=(1, n_class), dtype=tf.float32, initializer=tf.zeros_initializer())
init = tf.initializers.global_variables()

# Erstellung des Modells für Softmax-Klassifikation
z = tf.matmul(X, w) + b  # Liefert die eine Wahrscheinlichkeit zwischen 0 und 1
h = tf.nn.softmax(z)  # Berechnet die Hypothese (sagt quasi y vorher)

# Definition der Kostenfunktion und des Trainingsschrits (Minimierung Kostenfunktoin mittels Gradientenverfahren)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z,
                                                              name="Kostenfunktion"))  # Kreuzentropie liefert die Abweichung, muss noch mit 1/n multipliziert werden
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="Gradientenabstieg").minimize(loss)

# Berechnung der Vorhersagegenauigkeit
corr_preds = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
accuarcy = tf.reduce_sum(tf.cast(corr_preds, tf.float32))
# -----------------------------------------------------------------------------------------------------------------------


# Training des Modells---------------------------------------------------------------------------------------------------
writer = tf.summary.FileWriter('G:\Semester_7\ML_V6/graphs', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(init)  # Initialisierung
    n_batches = int(mnist.train.num_examples / batch_size)  # Berechnung Anzahl batches

    for k in range(n_epochs):
        ges_loss = 0

        for m in range(n_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, n_loss = sess.run([train_step, loss], feed_dict={X: x_batch, y: y_batch})
            ges_loss += n_loss
        print("loss in epoche{0}: {1}".format(k, (ges_loss / n_batches)))
    tf.Print(b, [b], "Wert für b:{0}".format([b]))
    tf.Print(w, [w], "Wert für w:{0}".format([w]))

    # Berechnung der Vorhersagegenauigkeit
    n_batches = int(mnist.test.num_examples / batch_size)
    corr_pred = 0

    for k in range(n_batches):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        n_accuary = sess.run(accuarcy, feed_dict={X: x_batch, y: y_batch})
        corr_pred += n_accuary
    print("Genauigkeit des Modells: {0}".format(corr_pred / mnist.test.num_examples))
writer.close()
# -----------------------------------------------------------------------------------------------------------------------
