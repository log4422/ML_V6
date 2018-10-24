#Aufgabe D4

#Bibliotheken importieren
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Variablendefinitionen
x = tf.placeholder(dtype=tf.float32)        #Platzhalter x
m = tf.constant(0.5, dtype=tf.float32)      #Konstante für m
t = tf.Variable(-2.0, name="t")             #Definierung Variable                           
n = 0                                       #Definition Zählvariable

y = m*x+t                                   #Definiiton des Berechnungsgraphen

y_np = np.zeros([600,5])                    #Definierung Ergebnismatrix


#Definition der Inkrementierung
assign_op = t.assign(t+1)


#Ausführen des Berechnungsgraphen
x_np = np.arange(-3.0,3.0,0.01)
with tf.Session() as sess:              #Oeffnen einer TensorFlow Session
    sess.run(t.initializer)                 #Initialisierung von t

    while n < 5:
        y_np[:,n] = sess.run(y, {x: x_np})      #Ausführen des Berechnungsgraphen
        print("t =",t.eval())                   #Ausgabe Wert von t   
        #print(y_np[:,0])                        #Ausgabe des Ergebnisses t (Startwert)
        sess.run(assign_op)                     #Inkrementierung von t
        n=n+1                                   #Inkrementierung von n
    #Schleifenende

    
    print("Form von y =",y_np.shape)                    #Ausgabe Shape des Ergebnisses

sess.close()                            #Schließen der Session


#Graphische Darstellung des Ergebnisses
plt.figure(1)
plt.plot(x_np,y_np[:,0],lw=2)
plt.plot(x_np,y_np[:,1],lw=2)
plt.plot(x_np,y_np[:,2],lw=2)
plt.plot(x_np,y_np[:,3],lw=2)
plt.plot(x_np,y_np[:,4],lw=2)
plt.grid(True)
plt.title("y=mx+t für t[-2,2]")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

