#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Autor: Lukas Götz
# Softmax-Regression, objektorientiert v1.0
# Datum: 29.10.2018
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Initialierungen von Tensorflow, Numpy und Matplotlib, Setup
import os
import tensorflow as tf
import time

# Deaktivierung von Warnungen
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import der MNIST Daten
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Definition der logits_pred Funktion
def logits_pred(X, n_classes, size_image, scope_name):
    """
    Funktion zur Vorhersage der Auftritswahrscheinlichkeiten
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        # Definition von weights and bias
        w = tf.get_variable(name="weights", shape=(size_image, n_classes), dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-5, 5))
        b = tf.get_variable(name="bias", shape=(1, n_classes), dtype=tf.float32,
                        initializer=tf.zeros_initializer())

        # Berechnung der Wahrscheinlichkeiten
        out = tf.matmul(X,w)+b
    return out
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Definition der  softmax_classifier Klasse
class softmax_classifier(object):

    def __init__(self):
        """
        Initialisierung der Klassen-Parameter
        """
        self.n_classes = 10
        self.size_image = 784
        self.batch_size = 128
        self.lr = 0.702
        self.print_step = 20
        #global_step wird vom Optimizer inkrementiert
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        #self.image  ->  Trainingsbild
        #self.label  ->  zugeöhrige Klasse

    def get_data(self):
        # Definiton von Platzhaltern X und y um das Modell mit Testdaten zu füttern
        self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.size_image), name="X")
        self.y = tf.placeholder(dtype=tf.int8, shape=(self.batch_size, self.n_classes), name="y")

    def prediction(self):
        """
        Berechnung der logids durch Aufruf der Funktion logids_pred
        """
        self.logits = logits_pred(self.X, self.n_classes, self.size_image, scope_name="fc_softmax_layer")

    def loss(self):
        """
        Berechnung der Abweichung zwischen Vorhersagen und tatsächlichen Werten
        """
        with tf.name_scope("loss"):
            # Kreuzentropie liefert die Abweichung, muss noch mit 1/n multipliziert werden
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits,
                                                                              name="Kreuzentropie"))
    def optimize(self):
        """
        Optimierung (Minimierung) der Kostenfunktion
        """
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr,
                                                     name="Gradientenabstieg").minimize(self.loss, global_step=self.gstep)

    def evaluate(self):
        """
        Berechnet die Vorhersagegenauigkeit
        """
        # Berechnet die Hypothese (sagt quasi y voraus)
        h = tf.nn.softmax(self.logits)

        # Berechnung der Vorhersagegenauigkeit
        corr_preds = tf.equal(tf.argmax(self.y, 1), tf.argmax(h, 1))
        self.accuracy = tf.reduce_sum(tf.cast(corr_preds, tf.float32))

    def summary(self):
        """
        Create summaries to write on TensorBoard
        """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        """
        Erstellen des Berechnungsgraph
        """
        self.get_data()
        self.prediction()
        self.loss()
        self.optimize()
        self.evaluate()
        self.summary()

#-----------------------------------------------------------------------------------------------------------------------

    def train_neo(self, n_epochs):
        """
        Trainieren des Modells
        """
        writer = tf.summary.FileWriter('G:\Semester_7\ML_V6\graphs/softmax_mnist', tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n_batches = int(mnist.train.num_examples / self.batch_size)  # Berechnung Anzahl batches
            #step = self.gstep.eval()

            for k in range(n_epochs):
                ges_loss = 0

                for m in range(n_batches):
                    x_batch, y_batch = mnist.train.next_batch(self.batch_size)
                    _, n_loss = sess.run([self.opt, self.loss], feed_dict={self.X: x_batch, self.y: y_batch})
                    ges_loss += n_loss

                print("loss in epoche{0}: {1}".format(k, (ges_loss / n_batches)))

            # Berechnung der Vorhersagegenauigkeit
            n_batches = int(mnist.test.num_examples / self.batch_size)
            corr_pred = 0

            for k in range(n_batches):
                x_batch, y_batch = mnist.test.next_batch(self.batch_size)
                n_accuary = sess.run(self.accuracy, feed_dict={self.X: x_batch, self.y: y_batch})
                corr_pred += n_accuary
            print("Genauigkeit des Modells: {0}".format(corr_pred / mnist.test.num_examples))
#            tf.Print(b, [b], "Wert für b:{0}".format([b]))
#            tf.Print(w, [w], "Wert für w:{0}".format([w]))

        writer.close()

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    Ausführen des Programms zur Softmax-Klassifikation des MNIST-Datensatzes
    """
    model = softmax_classifier()
    model.build()
    model.train_neo(n_epochs=300)
# ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
