#BIbliotheken importieren
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Erzeugen des Berechnungsgraphen
x = tf.placeholder(dtype=tf.float32) #Platzhalter x
y = x*x #Definition des Berechnungsgraphen

#Ausführen des Berechnungsgraphen
x_np = np.arange(-3.0,3.0,0.01)
sess = tf.Session() #Oeffnen einer TensorFlow Session
y_np =sess.run(y, {x: x_np}) #Ausführen des Berechnungsgraphen
print(y_np) #Ausgabe des Ergebnisses
print(y_np.shape) #Ausgabe des Ergebnisses
sess.close() #Schließen der Session

#Graphische Darstellung des Ergebnisses
plt.figure(1)
plt.plot(x_np, y_np, lw=2)
plt.grid(True)
plt.title('y=x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
