from Prepocessing import *
from developement import *
from validation_evaluation_strategies import *

filename = './dati/version_1.csv'
opener = scegli_opener(filename)

dati = DataCsv(opener,filename)
tupla = dati.load()

# Dataframe unico e pulito, da questo bisognerà dividere in training e set
data_unico = unificaDF(tupla[0],tupla[1])

#Una volta ottenuto training e set si trova la k ottima sul training

#Trovata la k ottima si esegue sul test e si calcolano le metriche





#Parte che non so se serve
classificatore=KNNClassifier(tupla[0],tupla[1])

print(f"\n K OTTIMALE")
print(f"K Ottimale trovato dal modello: k={classificatore.k}")
