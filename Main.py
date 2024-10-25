# Blocco 1: Installazione delle Dipendenze e Importazione delle Librerie

# Installa le dipendenze necessarie (esegui questi comandi nel terminale, non nel codice)
# pip install chembl_webresource_client rdkit-pypi scikit-learn seaborn psutil statsmodels

# Importa le librerie necessarie
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors

from chembl_webresource_client.new_client import new_client

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, ConfusionMatrixDisplay, r2_score
)
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer

from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import multiprocessing
import subprocess
import shlex
import time
import select
import psutil

# Blocco 2: Ricerca del Target e Download dei Dati di Bioattività

# Imposta il percorso base per salvare i file
base_path = r'C:\Users\marco\PycharmProjects\DrugRepurposing'

# Crea la cartella se non esiste
os.makedirs(base_path, exist_ok=True)

# Ricerca del target su ChEMBL
target = new_client.target
target_query = target.search('acetylcholinesterase')
targets = pd.DataFrame.from_dict(target_query)

# Filtra i target per 'Homo sapiens' o per un ChEMBL ID specifico
targets = targets[targets['organism'] == 'Homo sapiens']

if targets.empty:
    print("Errore: Nessun target trovato per 'acetylcholinesterase' in Homo sapiens.")
    sys.exit()

# Seleziona il target desiderato (ad esempio, il primo nella lista filtrata)
selected_target = targets.iloc[0]
target_chembl_id = selected_target['target_chembl_id']
print(f"Target selezionato: {selected_target['pref_name']} (ChEMBL ID: {target_chembl_id})")

# Percorso del file per i dati di bioattività
bioactivity_data_file = os.path.join(base_path, 'bioactivity_data_raw.csv')

# Verifica se il file esiste già
if os.path.exists(bioactivity_data_file):
    print(f"I dati di bioattività esistono già in {bioactivity_data_file}. Salto il download.")
    df = pd.read_csv(bioactivity_data_file)
else:
    # Download dei dati di bioattività (IC50) per il target selezionato
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_chembl_id, standard_type='IC50')

    # Converti i dati in un DataFrame
    df = pd.DataFrame.from_dict(res)

    # Salva i dati di bioattività in un file CSV
    df.to_csv(bioactivity_data_file, index=False)
    print(f"Dati di bioattività salvati in {bioactivity_data_file}")

# Blocco 3: Pre-elaborazione dei Dati di Bioattività

# Leggi i dati di bioattività
df = pd.read_csv(bioactivity_data_file)

# Seleziona le colonne di interesse
df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]

# Rimuovi duplicati basati su 'canonical_smiles'
df.drop_duplicates(subset='canonical_smiles', inplace=True)

# Rimuovi record con valori mancanti in 'canonical_smiles' o 'standard_value'
df.dropna(subset=['canonical_smiles', 'standard_value'], inplace=True)

# Converti 'standard_value' a numerico e gestisci errori
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')

# Rimuovi record con valori non numerici o mancanti in 'standard_value'
df.dropna(subset=['standard_value'], inplace=True)

# Filtra i dati per valori di 'standard_value' positivi
df = df[df['standard_value'] > 0]

# Visualizza la distribuzione dei valori di 'standard_value' usando una scala logaritmica
plt.figure(figsize=(8, 6))
sns.histplot(np.log10(df['standard_value']), bins=100)
plt.xlabel('Log10(Standard Value)')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Valori di Log10(IC50)')
plt.show()

# Identifica e gestisci gli outlier utilizzando l'Interquartile Range (IQR)
Q1 = df['standard_value'].quantile(0.25)
Q3 = df['standard_value'].quantile(0.75)
IQR = Q3 - Q1

# Definisci i limiti per gli outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtra i dati per rimuovere gli outlier
df = df[(df['standard_value'] >= lower_bound) & (df['standard_value'] <= upper_bound)].copy()

# Calcola il pIC50
df['pIC50'] = df['standard_value'].apply(lambda x: -np.log10(x * 1e-9))

# Visualizza la distribuzione dei valori di pIC50
plt.figure(figsize=(8, 6))
sns.histplot(df['pIC50'], bins=50, kde=True)
plt.xlabel('pIC50')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Valori di pIC50')
plt.show()

# Salva i dati pre-elaborati
preprocessed_data_file = os.path.join(base_path, 'bioactivity_data_preprocessed.csv')
df.to_csv(preprocessed_data_file, index=False)
print(f"Dati pre-elaborati salvati in {preprocessed_data_file}")

# Blocco 4: Calcolo dei Descrittori Molecolari con RDKit e Analisi Statistica

# Leggi i dati pre-elaborati
df = pd.read_csv(preprocessed_data_file)

# Lista dei descrittori da calcolare
descrittori = [
    'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'FractionCSP3',
    'BalabanJ', 'BertzCT', 'Chi0v', 'Kappa1', 'LabuteASA'
]

# Funzione per calcolare i descrittori con RDKit
def calcola_descrittori(dataframe):
    # Aggiungi una colonna 'Mol' con gli oggetti molecola di RDKit
    dataframe['Mol'] = dataframe['canonical_smiles'].apply(Chem.MolFromSmiles)

    # Rimuovi le molecole non valide (dove 'Mol' è None)
    dataframe = dataframe[dataframe['Mol'].notnull()].reset_index(drop=True)

    # Calcola i descrittori
    for desc in descrittori:
        func = getattr(Descriptors, desc, None)
        if func:
            dataframe[desc] = dataframe['Mol'].apply(func)
        else:
            print(f"Descrittore {desc} non trovato in RDKit.")

    # Rimuovi la colonna 'Mol' poiché non è più necessaria
    dataframe.drop(columns=['Mol'], inplace=True)

    return dataframe

# Calcola i descrittori
df = calcola_descrittori(df)

# Aggiungi una colonna 'Class' basata sul valore di pIC50
df['Class'] = df['pIC50'].apply(lambda x: 'Active' if x >= 6 else 'Inactive')

# Livello di significatività
alpha = 0.05

# Lista per memorizzare i risultati dell'analisi statistica
risultati = []

for desc in descrittori:
    # Valori per composti attivi e inattivi
    valori_attivi = df[df['Class'] == 'Active'][desc]
    valori_inattivi = df[df['Class'] == 'Inactive'][desc]

    # Esegui il test di Mann-Whitney U
    stat, p = mannwhitneyu(valori_attivi, valori_inattivi, alternative='two-sided')

    # Salva i risultati
    risultati.append({'Descrittore': desc, 'p-value': p})

# Converti i risultati in un DataFrame
risultati_df = pd.DataFrame(risultati)

# Applica la correzione per test multipli (FDR Benjamini-Hochberg)
reject, pvals_corrected, _, _ = multipletests(risultati_df['p-value'], alpha=alpha, method='fdr_bh')
risultati_df['p-value corretto'] = pvals_corrected
risultati_df['Significatività'] = ['Significativo' if r else 'Non significativo' for r in reject]

# Ordina i risultati per p-value corretto
risultati_df.sort_values('p-value corretto', inplace=True)

# Stampa la tabella dei risultati
print("\nTabella dei risultati con correzione per test multipli:")
print(risultati_df)

# Blocco 5: Calcolo dei Descrittori Estesi con PaDEL-Descriptor

# Percorso di output per i descrittori
descriptors_output_file = os.path.join(base_path, 'padel_descriptors.csv')

# Controlla se il file dei descrittori esiste già
if os.path.exists(descriptors_output_file) and os.path.getsize(descriptors_output_file) > 0:
    print("\nIl file dei descrittori esiste già. Salto il calcolo dei descrittori.")
else:
    # Verifica che il file PaDEL-Descriptor.jar esista nel percorso specificato
    jar_path = r'C:\Users\marco\PycharmProjects\DrugRepurposing\padel\PaDEL-Descriptor\PaDEL-Descriptor.jar'
    if not os.path.exists(jar_path):
        print("Errore: PaDEL-Descriptor.jar non trovato nel percorso specificato.")
        sys.exit("Interruzione dell'esecuzione a causa di un errore nel trovare PaDEL-Descriptor.jar.")

    # Leggi i dati pre-elaborati
    df = pd.read_csv(preprocessed_data_file)

    # Verifica che 'df' contenga 'molecule_chembl_id' e 'canonical_smiles'
    assert 'molecule_chembl_id' in df.columns
    assert 'canonical_smiles' in df.columns

    # Rimuovi record con SMILES o ID mancanti
    df.dropna(subset=['canonical_smiles', 'molecule_chembl_id'], inplace=True)

    # Crea un file SMILES per PaDEL-Descriptor
    smiles_file = os.path.join(base_path, 'molecules.smi')
    df[['canonical_smiles', 'molecule_chembl_id']].to_csv(
        smiles_file, sep='\t', header=False, index=False
    )

    # Imposta la memoria massima per Java e il numero di processori
    total_memory = psutil.virtual_memory().total  # Memoria totale in byte
    total_memory_gb = total_memory / (1024 ** 3)  # Converti in GB
    java_memory_value = int(total_memory_gb / 2)  # Metà della memoria totale
    java_memory = f'{java_memory_value}G'  # Formatta per Java

    print(f"Memoria totale disponibile: {total_memory_gb:.2f} GB")
    print(f"Memoria Java impostata a metà della memoria disponibile: {java_memory}")

    num_cpus = multiprocessing.cpu_count()  # Numero massimo di CPU disponibili
    print(f"Numero di CPU disponibili: {num_cpus}")

    # Comando per eseguire PaDEL-Descriptor
    cmd = f'java -Xms{java_memory} -Xmx{java_memory} -Djava.awt.headless=true -jar "{jar_path}" ' \
          f'-removesalt -standardizenitro -2d -fingerprints ' \
          f'-dir "{smiles_file}" -file "{descriptors_output_file}" -threads {num_cpus}'

    print(f"\nEsecuzione di PaDEL-Descriptor...")
    print(f"Comando eseguito:\n{cmd}\n")

    # Esegui il comando utilizzando subprocess.Popen
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print("Errori durante l'esecuzione di PaDEL-Descriptor:", stderr)

    except Exception as e:
        print(f"Errore durante l'esecuzione di PaDEL-Descriptor: {e}")
        process.kill()
        sys.exit("Interruzione dell'esecuzione a causa di un errore durante l'esecuzione di PaDEL-Descriptor.")

    # Verifica che il file dei descrittori sia stato creato correttamente
    if not os.path.exists(descriptors_output_file) or os.path.getsize(descriptors_output_file) == 0:
        print("\nErrore: il file dei descrittori non è stato creato correttamente o è vuoto.")
        sys.exit("Interruzione dell'esecuzione a causa di un errore nella creazione del file dei descrittori.")

    print("\nDescrittori calcolati con successo!")
    print(f"Descrittori salvati in: {descriptors_output_file}")


# # Blocco 5: Calcolo dei Descrittori Estesi con PaDEL-Descriptor
#
# # Importa le librerie necessarie
# import os
# import sys
# import pandas as pd
# import multiprocessing
# import subprocess
# import shlex
# import time
# import select
# import psutil  # Importazione di psutil per gestire la memoria
#
# # Percorso di output per i descrittori
# descriptors_output_file = os.path.join(base_path, 'padel_descriptors.csv')
#
# # Controlla se il file dei descrittori esiste già
# if os.path.exists(descriptors_output_file) and os.path.getsize(descriptors_output_file) > 0:
#     print("\nIl file dei descrittori esiste già. Salto il calcolo dei descrittori.")
# else:
#     # Verifica che il file PaDEL-Descriptor.jar esista nel percorso specificato
#     jar_path = os.path.join(base_path, 'padel/PaDEL-Descriptor/PaDEL-Descriptor.jar')
#     if not os.path.exists(jar_path):
#         print("Errore: PaDEL-Descriptor.jar non trovato nel percorso specificato.")
#         sys.exit("Interruzione dell'esecuzione a causa di un errore nel trovare PaDEL-Descriptor.jar.")
#
#     # Leggi i dati pre-elaborati
#     df = pd.read_csv(preprocessed_data_file)
#
#     # Verifica che 'df' contenga 'molecule_chembl_id' e 'canonical_smiles'
#     assert 'molecule_chembl_id' in df.columns
#     assert 'canonical_smiles' in df.columns
#
#     # Rimuovi record con SMILES o ID mancanti
#     df.dropna(subset=['canonical_smiles', 'molecule_chembl_id'], inplace=True)
#
#     # Imposta la memoria massima per Java e il numero di processori
#     total_memory = psutil.virtual_memory().total  # Memoria totale in byte
#     total_memory_gb = total_memory / (1024 ** 3)  # Converti in GB
#     java_memory_value = int(total_memory_gb / 2)  # Metà della memoria totale
#     java_memory = f'{java_memory_value}G'  # Formatta per Java
#
#     print(f"Memoria totale disponibile: {total_memory_gb:.2f} GB")
#     print(f"Memoria Java impostata a metà della memoria disponibile: {java_memory}")
#
#     num_cpus = multiprocessing.cpu_count()  # Numero massimo di CPU disponibili
#     print(f"Numero di CPU disponibili: {num_cpus}")
#
#     # Lista per tenere traccia degli ID delle molecole non processate
#     non_processati = []
#
#     # Timeout per singola molecola
#     timeout_per_molecule = 120  # Timeout in secondi per ogni SMILES
#
#     # Flag per indicare se è la prima iterazione (per gestire l'header)
#     prima_iterazione = True
#
#     # Ciclo per calcolare i descrittori per ciascun SMILES
#     for index, row in df.iterrows():
#         smiles = row['canonical_smiles']
#         molecule_id = row['molecule_chembl_id']
#
#         # Crea un file temporaneo per il singolo SMILES
#         temp_smiles_file = os.path.join(base_path, f"{molecule_id}_smiles.smi")
#         with open(temp_smiles_file, 'w') as file:
#             file.write(f"{smiles}\t{molecule_id}\n")
#
#         # Percorso di output temporaneo per i descrittori del singolo SMILES
#         temp_output_file = os.path.join(base_path, f"{molecule_id}_descriptors.csv")
#
#         # Comando per eseguire PaDEL-Descriptor su un singolo SMILES
#         cmd = f'/usr/bin/java -Xms{java_memory} -Xmx{java_memory} -Djava.awt.headless=true -jar "{jar_path}" ' \
#               f'-removesalt -standardizenitro -2d -fingerprints ' \
#               f'-dir "{temp_smiles_file}" -file "{temp_output_file}" -threads {num_cpus}'
#
#         print(f"\nEsecuzione di PaDEL-Descriptor per la molecola ID: {molecule_id}...")
#         print(f"Comando eseguito:\n{cmd}\n")
#
#         # Esegui il comando utilizzando subprocess.Popen
#         process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#
#         start_time = time.time()
#         try:
#             while True:
#                 # Controlla se il processo è terminato
#                 if process.poll() is not None:
#                     break
#
#                 # Controlla se il timeout è stato raggiunto
#                 elapsed_time = time.time() - start_time
#                 if elapsed_time > timeout_per_molecule:
#                     print(f"Il processo ha superato il tempo massimo di esecuzione per la molecola ID: {molecule_id} e sarà terminato.")
#                     process.kill()
#                     non_processati.append(molecule_id)
#                     break
#
#                 # Utilizza select per controllare se ci sono dati disponibili su stdout o stderr
#                 reads = [process.stdout.fileno(), process.stderr.fileno()]
#                 ret = select.select(reads, [], [], 1.0)  # Timeout di 1 secondo
#
#                 for fd in ret[0]:
#                     if fd == process.stdout.fileno():
#                         output = process.stdout.readline()
#                         if output:
#                             print(output.strip())
#                     elif fd == process.stderr.fileno():
#                         error = process.stderr.readline()
#                         if error:
#                             print("Errore:", error.strip())
#
#         except Exception as e:
#             print(f"Errore durante l'esecuzione per la molecola ID: {molecule_id}: {e}")
#             process.kill()
#             stdout, stderr = process.communicate()
#             if stdout:
#                 print(stdout)
#             if stderr:
#                 print("Errori durante l'esecuzione di PaDEL-Descriptor:", stderr)
#             non_processati.append(molecule_id)
#
#         finally:
#             # Assicurati che il processo sia terminato
#             process.wait()
#
#             # Leggi eventuali output residui
#             stdout, stderr = process.communicate()
#             if stdout:
#                 print(stdout)
#             if stderr:
#                 print("Errori durante l'esecuzione di PaDEL-Descriptor:", stderr)
#
#             # Rimuovi il file temporaneo del singolo SMILES
#             if os.path.exists(temp_smiles_file):
#                 os.remove(temp_smiles_file)
#
#         # Aggiungi i risultati al file complessivo dei descrittori
#         if os.path.exists(temp_output_file):
#             # Leggi il contenuto del file temporaneo
#             with open(temp_output_file, 'r') as temp_file:
#                 lines = temp_file.readlines()
#
#             # Se è la prima iterazione, scrivi tutto (header + dati)
#             if prima_iterazione:
#                 with open(descriptors_output_file, 'w') as out_file:
#                     out_file.writelines(lines)
#                 prima_iterazione = False
#             else:
#                 # Dalla seconda iterazione in poi, scrivi solo i dati (salta l'header)
#                 with open(descriptors_output_file, 'a') as out_file:
#                     out_file.writelines(lines[1:])  # Salta l'header
#
#             # Rimuovi il file temporaneo dei descrittori
#             os.remove(temp_output_file)
#
#     # Stampa un resoconto delle molecole non processate
#     if non_processati:
#         print(f"\nMolecole non processate a causa del timeout ({len(non_processati)} molecole):")
#         for molecule_id in non_processati:
#             print(f" - Molecule ID: {molecule_id}")
#     else:
#         print("Tutte le molecole sono state processate correttamente.")
#
#     # Verifica che il file dei descrittori sia stato creato correttamente
#     if not os.path.exists(descriptors_output_file) or os.path.getsize(descriptors_output_file) == 0:
#         print("\nErrore: il file dei descrittori non è stato creato correttamente o è vuoto.")
#         sys.exit("Interruzione dell'esecuzione a causa di un errore nella creazione del file dei descrittori.")
#
#     print("\nDescrittori calcolati con successo!")
#     print(f"Descrittori salvati in: {descriptors_output_file}")


# Blocco 6: Preparazione dei Dati per la Modellazione

# Percorso di output per i descrittori
descriptors_output_file = os.path.join(base_path, 'padel_descriptors.csv')

# Leggi i descrittori generati da PaDEL-Descriptor
try:
    df_descriptors = pd.read_csv(
        descriptors_output_file,
        sep=',',
        header=0,
        index_col=None,
        engine='python',
        on_bad_lines='skip'
    )
except FileNotFoundError:
    print(f"Errore: Il file {descriptors_output_file} non esiste.")
    sys.exit("Assicurati che il file dei descrittori sia stato generato correttamente.")

# Rinomina la colonna degli ID molecolari se necessario
if 'Name' in df_descriptors.columns:
    df_descriptors.rename(columns={'Name': 'molecule_chembl_id'}, inplace=True)
else:
    print("Errore: La colonna 'Name' non è presente in df_descriptors.")
    sys.exit("Interruzione dell'esecuzione a causa di un errore nel file dei descrittori.")

# Pulizia degli ID molecolari
df_descriptors['molecule_chembl_id'] = df_descriptors['molecule_chembl_id'].astype(str).str.strip().str.upper()

# Leggi df_pic50
preprocessed_data_file = os.path.join(base_path, 'bioactivity_data_preprocessed.csv')
try:
    df_pic50 = pd.read_csv(preprocessed_data_file)
except FileNotFoundError:
    print(f"Errore: Il file {preprocessed_data_file} non esiste.")
    sys.exit("Assicurati che il file pre-elaborato sia stato generato correttamente.")

# Pulizia degli ID molecolari
df_pic50['molecule_chembl_id'] = df_pic50['molecule_chembl_id'].astype(str).str.strip().str.upper()

# Merge dei due DataFrame
df_merged = pd.merge(
    df_pic50[['molecule_chembl_id', 'pIC50']],
    df_descriptors,
    on='molecule_chembl_id',
    how='inner'
)

print(f"Numero di composti dopo il merge: {df_merged.shape[0]}")

# Gestione dei valori mancanti
df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

# Rimuovi colonne con più del 50% di valori mancanti
df_merged.dropna(axis=1, thresh=int(0.5 * df_merged.shape[0]), inplace=True)

# Imputazione dei valori mancanti con la mediana
imputer = SimpleImputer(strategy='median')
imputed_data = imputer.fit_transform(df_merged.drop(columns=['molecule_chembl_id', 'pIC50']))

# Creazione di un DataFrame con i dati imputati
X = pd.DataFrame(imputed_data, columns=df_merged.drop(columns=['molecule_chembl_id', 'pIC50']).columns)

# Target Y
Y = df_merged['pIC50'].reset_index(drop=True)

# Rimozione delle caratteristiche con varianza zero
selector = VarianceThreshold(threshold=0.0)
X_reduced = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]

# Converti X_reduced in DataFrame per mantenere i nomi delle colonne
X_reduced = pd.DataFrame(X_reduced, columns=selected_features)

print(f"\nDimensione finale della matrice delle caratteristiche dopo la rimozione delle caratteristiche con varianza zero: {X_reduced.shape}")

# Blocco 7: Miglioramento delle Performance del Modello

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
import joblib  # Per il salvataggio dei modelli
from scipy.stats import uniform, randint

# Assicurati di avere X_reduced e Y dai blocchi precedenti

# --- Preprocessing dei Dati ---

# 1. Rimuovi caratteristiche con varianza molto bassa
selector_variance = VarianceThreshold(threshold=1e-5)
X_high_variance = selector_variance.fit_transform(X_reduced)

# 2. Imputazione dei valori mancanti con la mediana
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_high_variance)

# 3. Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4. Verifica di eventuali valori NaN o infiniti in X_scaled
if np.isnan(X_scaled).sum() > 0 or np.isinf(X_scaled).sum() > 0:
    print("Attenzione: Ci sono valori NaN o infiniti in X_scaled.")
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
else:
    print("Nessun valore NaN o infinito in X_scaled.")

# 5. Suddivisione dei dati in training e test set
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# 6. Verifica di eventuali valori NaN o infiniti in y_train_reg
if np.isnan(y_train_reg).sum() > 0 or np.isinf(y_train_reg).sum() > 0:
    print("Attenzione: Ci sono valori NaN o infiniti in y_train_reg.")
    y_train_reg = np.nan_to_num(y_train_reg, nan=0.0, posinf=0.0, neginf=0.0)
else:
    print("Nessun valore NaN o infinito in y_train_reg.")

# --- Selezione delle Caratteristiche con SelectFromModel ---

# 7. Utilizziamo RandomForestRegressor per la selezione delle caratteristiche
feature_selector_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
feature_selector = SelectFromModel(
    estimator=feature_selector_model,
    threshold='median'
)

feature_selector.fit(X_train, y_train_reg)

X_train_sel = feature_selector.transform(X_train)
X_test_sel = feature_selector.transform(X_test)

print(f"Numero di caratteristiche selezionate: {X_train_sel.shape[1]}")

# --- Pipeline per la Regressione e Classificazione ---

# Definisci una pipeline per il preprocessing e il modello
pipeline_rf = Pipeline([
    ('feature_selector', SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold='median'
    )),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

pipeline_gb = Pipeline([
    ('feature_selector', SelectFromModel(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        threshold='median'
    )),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# --- Ottimizzazione degli Iperparametri con RandomizedSearchCV ---

# Definizione della griglia di parametri per RandomForestRegressor
param_distributions_rf = {
    'regressor__n_estimators': randint(100, 300),
    'regressor__max_depth': [5, 10, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2', None]
}

# Randomized Search per RandomForestRegressor
random_search_rf = RandomizedSearchCV(
    estimator=pipeline_rf,
    param_distributions=param_distributions_rf,
    n_iter=50,  # Numero di combinazioni da provare
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)

# Adattamento del modello
random_search_rf.fit(X_train, y_train_reg)

# Migliori parametri trovati
print("Migliori parametri per Random Forest Regressor:")
print(random_search_rf.best_params_)

# Valutazione sul training set
y_train_pred_rf = random_search_rf.predict(X_train)
r2_train_rf = r2_score(y_train_reg, y_train_pred_rf)
mse_train_rf = mean_squared_error(y_train_reg, y_train_pred_rf)
print(f"R2 Score sul training set con Random Forest: {r2_train_rf:.3f}")
print(f"MSE sul training set con Random Forest: {mse_train_rf:.3f}")

# Valutazione sul test set
y_test_pred_rf = random_search_rf.predict(X_test)
r2_test_rf = r2_score(y_test_reg, y_test_pred_rf)
mse_test_rf = mean_squared_error(y_test_reg, y_test_pred_rf)
print(f"R2 Score sul test set con Random Forest: {r2_test_rf:.3f}")
print(f"MSE sul test set con Random Forest: {mse_test_rf:.3f}")

# Valutazione con Cross-Validation
cv_scores_rf = cross_val_score(
    random_search_rf.best_estimator_,
    X_scaled,
    Y,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
print(f"R2 Score medio con cross-validation (Random Forest): {np.mean(cv_scores_rf):.3f}")

# --- Modello di Classificazione con Random Forest ---

# Creazione della variabile target binaria
y_train_clf = pd.Series(y_train_reg).apply(lambda x: 1 if x >= 6 else 0)
y_test_clf = pd.Series(y_test_reg).apply(lambda x: 1 if x >= 6 else 0)

# Definizione della pipeline per la classificazione
pipeline_rf_clf = Pipeline([
    ('feature_selector', SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold='median'
    )),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Definizione della griglia di parametri per RandomForestClassifier
param_distributions_rf_clf = {
    'classifier__n_estimators': randint(100, 300),
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Randomized Search per RandomForestClassifier
random_search_rf_clf = RandomizedSearchCV(
    estimator=pipeline_rf_clf,
    param_distributions=param_distributions_rf_clf,
    n_iter=50,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

# Adattamento del modello
random_search_rf_clf.fit(X_train, y_train_clf)

# Migliori parametri trovati
print("\nMigliori parametri per Random Forest Classifier:")
print(random_search_rf_clf.best_params_)

# Valutazione sul training set
y_train_pred_rf_clf = random_search_rf_clf.predict(X_train)
accuracy_train_rf = accuracy_score(y_train_clf, y_train_pred_rf_clf)
print(f"Accuracy sul training set con Random Forest Classifier: {accuracy_train_rf:.3f}")

# Report di classificazione sul training set
print("\nReport di classificazione sul training set (Random Forest Classifier):")
print(classification_report(y_train_clf, y_train_pred_rf_clf, target_names=['Inactive', 'Active']))

# Matrice di confusione sul training set
cm_train_rf = confusion_matrix(y_train_clf, y_train_pred_rf_clf)
disp_train_rf = ConfusionMatrixDisplay(confusion_matrix=cm_train_rf, display_labels=['Inactive', 'Active'])
disp_train_rf.plot()
plt.title('Confusion Matrix - Training Set (Random Forest Classifier)')
plt.show()

# Valutazione sul test set
y_test_pred_rf_clf = random_search_rf_clf.predict(X_test)
accuracy_test_rf = accuracy_score(y_test_clf, y_test_pred_rf_clf)
print(f"Accuracy sul test set con Random Forest Classifier: {accuracy_test_rf:.3f}")

# Report di classificazione sul test set
print("\nReport di classificazione sul test set (Random Forest Classifier):")
print(classification_report(y_test_clf, y_test_pred_rf_clf, target_names=['Inactive', 'Active']))

# Matrice di confusione sul test set
cm_test_rf = confusion_matrix(y_test_clf, y_test_pred_rf_clf)
disp_test_rf = ConfusionMatrixDisplay(confusion_matrix=cm_test_rf, display_labels=['Inactive', 'Active'])
disp_test_rf.plot()
plt.title('Confusion Matrix - Test Set (Random Forest Classifier)')
plt.show()

# Valutazione con Cross-Validation
cv_scores_rf_clf = cross_val_score(
    random_search_rf_clf.best_estimator_,
    X_scaled,
    y_train_clf.append(y_test_clf),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
print(f"Accuracy media con cross-validation (Random Forest Classifier): {np.mean(cv_scores_rf_clf):.3f}")

# --- Analisi delle Importanze delle Caratteristiche ---

# Analisi delle feature importances dal modello Random Forest Regressor
importances = random_search_rf.best_estimator_.named_steps['regressor'].feature_importances_
feature_names = np.array(X_reduced.columns)[selector_variance.get_support()]

# Creare un DataFrame per visualizzare le importanze
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Ordinare le feature per importanza decrescente
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Visualizzare le prime 10 caratteristiche più importanti
print("\nCaratteristiche più importanti secondo il Random Forest Regressor:")
print(feature_importances.head(10))

# Grafico delle importanze
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title('Top 10 Feature Importances')
plt.show()

# Salva i modelli migliori per uso futuro (opzionale)
joblib.dump(random_search_rf.best_estimator_, os.path.join(base_path, 'best_random_forest_regressor.pkl'))
joblib.dump(random_search_rf_clf.best_estimator_, os.path.join(base_path, 'best_random_forest_classifier.pkl'))
