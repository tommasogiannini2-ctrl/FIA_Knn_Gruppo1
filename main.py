from Prepocessing import *
from developement import *
from validation_evaluation_strategies import *

filename = './dati/version_1.csv'
opener = scegli_opener(filename)

dati = Data(opener,filename)
tupla = dati.load()

# Dataframe unico e pulito, da questo bisognerà dividere in training e set
data_unico = unificaDF(tupla[0],tupla[1])
#print(data_unico)
# Parametri per la divisione
n_prove = 5
data = ValidationStrategy(data_unico)
p_RandomSubsampling = 0.8
p_Holdout = 0.8

# Divisione con holdout
lista_holdout = data.RandomSubsampling(1,p_Holdout)
# Questa lista contiene una coppia di training e test divise secondo il metodo Holduot

# Divisione training e set con il metodo Kfold
lista_Kfold = data.Kfold(n_prove)
# Questa lista contiene n coppie di training e test divise secondo il metodo Kfold
"""
Ciclo per vedere se funziona tutto
for i in range(n_prove):
    coppia = lista_Kfold[i]
    print(f"coppia {i + 1}")
    print(coppia[0].info())
    print(coppia[1].info())
"""

# Divisione training e set con il metodo RandomSubsampling
lista_RS = data.RandomSubsampling(n_prove, p_RandomSubsampling)
# Questa lista contiene n coppie di training e test divise secondo il metodo Random Subsampling
"""
Ciclo per vedere se funziona tutto
for i in range(n_prove):
    coppia = lista_RS[i]
    print(f"coppia {i + 1}")
    print(coppia[0].info())
    print(coppia[1].info())
"""


#Una volta ottenuto training e set si trova la k ottima sul training

#Trovata la k ottima si esegue sul test e si calcolano le metriche


#Parte che non so se serve
#classificatore=KNNClassifier(tupla[0],tupla[1])

#print(f"\n K OTTIMALE")
#print(f"K Ottimale trovato dal modello: k={classificatore.k}")
