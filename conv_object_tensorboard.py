#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Autor: Lukas Götz
# ConvNet für den MNIST-Datensatz mit Tensorboard-Visualisierung
# Datum: 07.11.2018
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
def conv_layer(inputs ,filters, k_size, stride, padding, scope_name):
    """"
    Durchführung der Faltung, Anwendung der Aktivierungsfunkiton
    """
    if scope_name == "conv1":
        inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:


        in_chanels = inputs.shape[-1]
        kernel = tf.get_variable("kernel",
                                 [k_size, k_size, in_chanels, filters],
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable("biases",
                                 [filters],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv + biases, name="relu")

#-----------------------------------------------------------------------------------------------------------------------

def max_pool(inputs, ksize, stride, padding="VALID", scope_name="pool"):
    """
    Methode für Maxpooling
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
        return pool

#-----------------------------------------------------------------------------------------------------------------------

def fully_connection(inputs, out_dim,  scope_name="fc"):
    """
    Funktion zur Vorhersage der Auftritswahrscheinlichkeiten
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        # Definition von weights and bias
        w = tf.get_variable(name="weights", shape=(in_dim, out_dim), dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-5, 5))
        b = tf.get_variable(name="bias", shape=(1, out_dim), dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        # Berechnung der Wahrscheinlichkeiten
        out = tf.matmul(inputs,w)+b

    return out
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Definition der  softmax_classifier Klasse
class convNet(object):

    def __init__(self):
        """
        Initialisierung der Klassen-Parameter
        """
        self.n_classes = 10
        self.size_image = 784
        self.batch_size = 128
        self.lr = 0.001
        self.keep_prob = tf.constant(0.75)
        self.print_step = 100
        #global_step wird vom Optimizer inkrementiert
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def get_data(self):
        # Definiton von Platzhaltern X und y um das Modell mit Testdaten zu füttern
        self.X = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.size_image), name="X")
        self.y = tf.placeholder(dtype=tf.int8, shape=(self.batch_size, self.n_classes), name="y")

    def prediction(self):
        """
        Festlegung der Netzwerkstruktur
        """
        print("self.X: ", self.X)
        conv1 = conv_layer(inputs=self.X,
                           filters=32,
                           k_size=5,
                           stride=1,
                           padding="SAME",
                           scope_name="conv1")
        print("conv1: ", conv1)
        pool1 = max_pool(conv1, 2, 2, "VALID", "pool1")
        print("pool1: ", pool1)
        conv2 = conv_layer(inputs=pool1,
                           filters=64,
                           k_size=5,
                           stride=1,
                           padding="SAME",
                           scope_name="conv2")
        print("conv2: ", conv2)
        pool2 = max_pool(conv2, 2, 2, "VALID", "pool2")
        print("pool2: ", pool2)
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        print("feature_dim: ",feature_dim)
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        print("pool2_neo: ", pool2)
        fc =fully_connection(pool2, 1024 , scope_name="fc")

        with tf.name_scope("regularisation") as scope:
            dropout = tf.nn.dropout(tf.nn.relu(fc), self.keep_prob, name="relu_dropout")


        self.logits = fully_connection(dropout, self.n_classes, scope_name="logits")

    def loss(self):
        """
        Berechnung der Abweichung zwischen Vorhersagen und tatsächlichen Werten
        """
        with tf.name_scope("loss"):
            # Kreuzentropie liefert die Abweichung, muss noch mit 1/n multipliziert werden
            print("self.logits: ", self.logits)
            print("self.y: ", self.y)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits,
                                                                              name="Kreuzentropie"))
    def optimize(self):
        """
        Optimierung (Minimierung) der Kostenfunktion
        """
        with tf.name_scope("optimize"):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.gstep)

    def evaluate(self):
        """
        Berechnet die Vorhersagegenauigkeit
        """
        # Berechnet die Hypothese (sagt quasi y voraus)
        with tf.name_scope("prediciton"):
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
            #tf.summary.histogram("histogram loss", self.loss)
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
        #Initialisierung Startzeit, Definiton FileWriter
        start_zeit = time.time()
        writer = tf.summary.FileWriter('G:\Semester_7\ML_V6\graphs/softmax_mnist', tf.get_default_graph())

        with tf.Session() as sess:

            #Initialisierung der Variablen
            sess.run(tf.global_variables_initializer())

            #Parameter Definition n_batchds, step-Variable
            n_batches = int(mnist.train.num_examples / self.batch_size)  # Berechnung Anzahl batches
            step = 0
            print("\nEs werden pro Epoche {0} batches a 128 Bildern verarbeitet.\n".format(n_batches))

            #Trainieren des Modells
            for k in range(n_epochs):
                ges_loss = 0
                for m in range(n_batches):
                    x_batch, y_batch = mnist.train.next_batch(self.batch_size)
                    _, n_loss, summaries = sess.run([self.opt, self.loss, self.summary_op], feed_dict={self.X: x_batch, self.y: y_batch})
                    if (step%self.print_step) == 0:
                        writer.add_summary(summaries, global_step=step)
                    ges_loss += n_loss
                    step += 1
                print("loss in epoche{0}: {1}".format(k, (ges_loss / n_batches)))

            # Berechnung der Vorhersagegenauigkeit
            n_batches = int(mnist.test.num_examples / self.batch_size)
            corr_pred = 0
            for k in range(n_batches):
                x_batch, y_batch = mnist.test.next_batch(self.batch_size)
                n_accuary = sess.run(self.accuracy, feed_dict={self.X: x_batch, self.y: y_batch})
                corr_pred += n_accuary
            print("\nGenauigkeit des Modells: {0}".format(corr_pred / mnist.test.num_examples))
            print("\nEs wurden insgesamt Trainingsschritte (a 128 Bilder) durchlaufen:")
            print(sess.run(self.gstep))
        writer.close()

        # Ausgabe der Programmlaufzeit
        end_zeit = time.time()
        laufzeit = end_zeit-start_zeit
        print("\nDas Training des Models benötigte {0:2.2f} Sekunden".format(laufzeit))

        print(int(mnist.train.num_examples/1))
        print(int(mnist.test.num_examples/1))
        print(int(mnist.validation.num_examples/1))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    Ausführen des Programms zur Softmax-Klassifikation des MNIST-Datensatzes
    """
    model = convNet()
    model.build()                   # Erstellen eines Objekts convNet
    model.train_neo(n_epochs=10)    # Trainieren des Modells mit n-Epochen
# ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
