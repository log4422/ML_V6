#Aufgabe D2

import numpy as np
import math

#Unterprogramm zur Berechnung der Rechteckspezifikationen
def rechteck(a,b):
    A=a*b
    U=2*a+2*b
    dl2=a*a+b*b
    dl=math.sqrt(dl2)
    return A,U,dl


#Hauptprogramm
nochmal='ja'
while nochmal == 'ja':
    a=int(input('\nBitte geben Sie die Länge des Rechtecks ein: '))
    b=int(input('Bitte geben Sie die Breite des Rechtecks ein: '))

    A,U,dl = rechteck(a,b)

    print(f'\nFlächeninhalt: {A:.2f}')
    print(f'Umfang: {U:.2f}')
    print(f'Diagonallänge: {dl:.2f}')

    print('\nWeiteres Rechteck berechnen (ja/nein)?')
    nochmal=input()
