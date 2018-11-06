#Machine Learning Versuch 6
#Aufgabe D3
#Autor: Lukas Götz
#Datum: 13.10.2018


#-----------------------------------------------------------------------------------------
#Anmerkung: 100% korrekt ist der Versuch in Matlab hinterlegt
#G:\Semester_6\ML\ML_Klausur\V2_
#D2_Vorlage.m
#-----------------------------------------------------------------------------------------



# Bibliotheken importieren ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Daten importieren und Variablen extrahieren
df = pd.read_excel('Autos_DE.xlsx')
DATA = df.values
y = np.array(DATA[2:,0], dtype='float') #Verbrauch
x1 = np.array(DATA[2:,1], dtype='float') #Zylinderanzahl
x2 = np.array(DATA[2:,2], dtype='float') #Hubraum
x3 = np.array(DATA[2:,3], dtype='float') #Leistung
x4 = np.array(DATA[2:,4], dtype='float') #Gewicht
x5 = np.array(DATA[2:,5], dtype='float') #Beschleunigung
x6 = np.array(DATA[2:,6], dtype='float') #Baujahr
m = y.shape[0]
x0 = np.array(np.ones(m), dtype='float') #Hilfsvariable bestehend aus Einsen
X = np.stack((x0,x1,x2,x3,x4,x5,x6)).T   #Datenmatrix X
#-----------------------------------------------------------------------------------------


# Berechnung der Regressionsparameter für alle Eingansvariablen---------------------------
Rxx = np.dot(X.T,X)                      #Korrelationsmatrix
beta = np.dot(np.dot(np.linalg.inv(Rxx),X.T),y) #beta-Vektor
print('Parametervektor (mit allen Eingangsvariablen) beta=', beta)
#-----------------------------------------------------------------------------------------



# Scatterplot der Daten Hubraum-----------------------------------------------------------
plt.figure(1)
plt.scatter(x2,y,s=3,c='red')
plt.grid(True)
plt.title('Scatterplot')
plt.xlabel('Hubraum x2 in Liter')
plt.ylabel('Verbrauch  in Liter/100km')
plt.show(block=False)  #block=False: Skript laeuft weiter

#Berechnung der Regressionsparameter Hubraum einzeln		
x0 = np.array(np.ones(m), dtype='float') #Hilfsvariable bestehend aus Einsen	
X2 = np.stack((x0,x2)).T #Datenmatrix X
Rxx2  = np.dot(X2.T,X2) #Korrelationsmatrix
beta2 = np.dot(np.dot(np.linalg.inv(Rxx2),X2.T),y)
print("Parametervektor (für Hubraum) beta2= ", beta2)

#Graphische Darstellung der Regressionsgerade Hubraum
x_2 = np.linspace(np.min(x2),np.max(x2),200) #Hilfsvariable x_
y_2 = beta2[0]+beta2[1]*x_2 #Hilfsvariable y_
plt.plot(x_2,y_2,lw=2)
plt.grid(True)
plt.title('Datenpunkte mit Regressionsgerade')
plt.xlabel('Hubraum x2 in Liter')
plt.ylabel('Verbrauch  in Liter/100km')
plt.show(block=False)

# Scatterplot der Daten Gewicht
plt.figure(2)
plt.scatter(x4,y,s=3,c='red')
plt.grid(True)
plt.title('Scatterplot')
plt.xlabel('Gewicht x3 in kg')
plt.ylabel('Verbrauch in Liter/100km')
plt.show(block=False)  #block=False: Skript laeuft weiter

#Berechnung der Regressionsparameter Gewicht einzeln		
x0 = np.array(np.ones(m), dtype='float') #Hilfsvariable bestehend aus Einsen	
X4 = np.stack((x0,x4)).T #Datenmatrix X
Rxx4  = np.dot(X4.T,X4) #Korrelationsmatrix
beta4 = np.dot(np.dot(np.linalg.inv(Rxx4),X4.T),y)
print("Parametervektor (für Gewicht) beta4= ", beta4)

#Graphische Darstellung der Regressionsgerade Gewicht
x_4 =np.linspace(np.min(x4),np.max(x4),200) #Hilfsvariable x_
y_4 = beta4[0]+beta4[1]*x_4 #Hilfsvariable y_
plt.plot(x_4,y_4,lw=2)
plt.grid(True)
plt.title('Datenpunkte mit Regressionsgerade')
plt.xlabel('Gewicht x3 in kg')
plt.ylabel('Verbrauch in Liter/100km')
plt.show(block=False)

plt.show() #Verhindert, dass Grafikfenster sofort geschlossen wird
#-----------------------------------------------------------------------------------------
