import pandas as pd

# Sottoprogramma apertura file csv
def apertura_file_csv(nome_file_csv: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(nome_file_csv)
        print(f"File '{nome_file_csv}' estratto con successo nel DataFrame 'df'.\n")

        print("\n--- Informazioni sulla struttura del DataFrame (df.info()) ---")
        df.info()
    except FileNotFoundError:
        print(f"ERRORE: Il file '{nome_file_csv}' non è stato trovato. Assicurati che sia nella stessa cartella dello script Python o che il percorso sia corretto.")
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")
    return df

if __name__ == '__main__':
    nf = "./dati/version_1.csv"
    dati = apertura_file_csv(nf)