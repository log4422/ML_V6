#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Autor: Lukas Götz
# Softmax-Regression, objektorientiert für den MNIST-Datensatz
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Definition der logits_pred Funktion
def logits_pred(X, n_classes, size_image, scope_name):
    """
    Funktion zur Vorhersage der Auftritswahrscheinlichkeiten
    Hier eine einfache Softmax-Regression
    """
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

    def get_data(self):
        """
        Definiton von Platzhaltern X und y um das Modell mit Testdaten befüllen zu können
        """

    def prediction(self):
        """
        Berechnung der logits durch Aufruf der ausgelagerten Funktionen (bzw. Struktur)
        """

    def loss(self):
        """
        Berechnung der Abweichung zwischen Vorhersagen und tatsächlichen Werten
        """
    def optimize(self):
        """
        Optimierung (Minimierung) der Kostenfunktion
        """

    def evaluate(self):
        """
        Berechnet die Vorhersagegenauigkeit
        """

    def summary(self):
        """
        Erstellen einer Zusammenfassung/Summary für Tensorboard
        """

    def build(self):
        """
        Erstellen des Berechnungsgraphen
        """

    def train_neo(self, n_epochs):
        """
        Trainieren des Modells mit anschließender Überprüfung der Vorhersagegenauigkeit
        """
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    Ausführen des Programms zur Softmax-Klassifikation (des MNIST-Datensatzes)
    """
    model = softmax_classifier()
    model.build()
    model.train_neo(n_epochs=400)
# ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
