#  Progetto FIA - Classificatore di Tumori k-NN (Gruppo 1): Giannini Tommaso, di Raimondo Gabriele, Romito Flavia



## Obiettivo del Progetto

Il progetto mira a sviluppare una \*\*pipeline di Machine Learning\*\* generica in Python che addestra e valuta un classificatore \*\*k-Nearest Neighbors (k-NN)\*\* . Il modello è utilizzato per classificare le cellule tumorali come benigne (Classe 2) o maligne (Classe 4) in base a 9 caratteristiche fornite.

---



## Configurazione e Specifiche Tecniche



### Parametri Definitivi del Gruppo (i=1)

| \*\*A (Input Data)\*\* | `version\_1.csv` | I dati di input devono essere caricati da un file in formato \*\*CSV\*\*. |

| \*\*B (Validazione 1)\*\* | \*\*`Random Subsampling`\*\* | Tecnica di valutazione richiesta in modalità \*K\* esperimenti. |

| \*\*C (Validazione 2)\*\* | \*\*`K-fold Cross Validation`\*\* | Seconda tecnica di valutazione richiesta in modalità \*K\* esperimenti (deve gestire le folds). |



### Modularizzazione e Pattern

* Il codice include la \*\*verifica automatica di almeno 3 test di correttezza\*\* tramite il modulo `unittest`.



### Dipendenze

Sono consentiti e utilizzati i seguenti pacchetti non standard:

* `numpy` (per operazioni vettoriali e matriciali).

* `pandas` (per il caricamento e la manipolazione del dataset).

* `matplotlib` (per la visualizzazione dei risultati, es. Confusion Matrix, ROC).

*'ABC' per l'implemetazione delle factory.

aggiungi eventuali altre

---


## Dataset e Data Preprocessing
I dati iniziali letti da un file csv sono stati elaborati secondo i seguenti passaggi: controllo ed eliminazione dei duplicati, eliminazioni features superficiali, eliminazione dei dati nulli, eliminazione righe che hanno valore nulli nella colonna 'classtype_v1',eliminazione righe con più di 4 valori nulli, eliminazione dei dati fuori dal range 1-10 e 2/4 per classtype_v1, conversione stringhe in interi.
(da completare)

### Struttura del Dataset

Il dataset è un file di tipo .csv. Attraverso le factory è concessa anche la lettura di file con estensione XLS e SQL.

Il dataset contiene le seguenti colonne:

* \*\*ID/Identificativi:\*\* `ID`, `Sample code number` (Non utilizzate come features).

* \*\*Features (X):\*\* 9 colonne con valori ordinali interi da 1 a 10.

* \*\*Target (Y):\*\* `Class` (2 = Benigno, 4 = Maligno).



### Gestione delle Peculiarità (Decisione Implementativa)

Nel dataset è attesa la presenza di valori mancanti, contrassegnati dal carattere \*\*<<Null>>\*\*, o anomali.
i Valori nulli sono risolti con sostituzione della moda perchè ritenuta più rappresentativa in quanto mostra il valore più probabile senza necessità di arrotondamento del valore. 


*\*Strategia Adottata (Da Completare):\*\* 



---



## Istruzioni per l'Esecuzione



### 1. Preparazione dell'Ambiente



1\.  Clonare il repository GitHub:

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/tommasogiannini2-ctrl/FIA\_Knn\_Gruppo1.git](https://github.com/tommasogiannini2-ctrl/FIA\_Knn\_Gruppo1.git)

&nbsp;   cd FIA\_Knn\_Gruppo1

&nbsp;   ```

2\.  Creare e attivare l'ambiente virtuale:

&nbsp;   ```bash

&nbsp;   # (Se non si usa PyCharm)

&nbsp;   source env/Scripts/activate 

&nbsp;   ```

3\.  Installare le dipendenze:

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```

4\.  Inserire il file \*\*`version\_1.csv`\*\* nella cartella \*\*`data/`\*\*.



### 2. Esecuzione del Programma (Da Definire)



Il programma dovrà essere eseguito tramite il file principale `main.py` specificando i parametri richiesti dall'utente.

(da implementare)



---



## Risultati



I file di output verranno salvati nella directory \*\*`results/`\*\*.



\*\*\[AGGIUNGERE QUI LA GUIDA ALL'INTERPRETAZIONE DEI RISULTATI]\*\*

