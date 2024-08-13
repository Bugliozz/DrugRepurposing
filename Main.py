import os
import subprocess
from subprocess import PIPE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Configura Pandas per visualizzare tutte le colonne e le righe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("\n")

# 1) RICERCA DEI TARGET
search_term = 'acetylcholinesterase'
target = new_client.target
target_query = target.search(search_term)
targets = pd.DataFrame.from_dict(target_query)
print("PER LA RICERCA DEI TARGET DI " + search_term.upper() + " SONO STATI TROVATI I SEGUENTI RISULTATI: \n" + targets.to_string() + "\n")

# 2) SELEZIONARE E RECUPERARE I DATI DI BIOATTIVITÀ
numero = int(input("Inserisci il numero del target: "))
selected_target = targets.target_chembl_id[numero]
print("\nL'ID SELEZIONATO è: " + selected_target + "\n")
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
filename = search_term + '_bioactivity_data.csv'
if not os.path.isfile(filename):
    df.to_csv(filename, index=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# 3) PRE-ELABORAZIONE DEI DATI SULLA BIOATTIVITÀ E CREAZIONE DEI FILE CSV
def norm_value(input):
    norm = []
    for i in input['standard_value']:
        i = float(i)
        if i > 100000000:
            i = 100000000
        norm.append(i)
    input['standard_value_norm'] = norm
    input = input.drop('standard_value', axis=1)
    return input

def pIC50(input, threshold_active=6, threshold_inactive=5):
    pIC50 = []
    min_value = 1e-10
    for i in input['standard_value_norm']:
        i = float(i)
        if i == 0:
            i = min_value
        molar = i * (10 ** -9)
        pIC50_value = -np.log10(molar)
        pIC50.append(pIC50_value)
    input['pIC50'] = pIC50
    input = input.drop('standard_value_norm', axis=1)
    return input

def lipinski(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        if mol is not None:
            moldata.append(mol)
        else:
            moldata.append(None)

    baseData = []
    for mol in moldata:
        if mol is not None:
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
            row = [desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]
        else:
            row = [np.nan, np.nan, np.nan, np.nan]
        baseData.append(row)

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(baseData, columns=columnNames)
    return descriptors

# Seleziona i dati dal DataFrame grezzo
selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df2 = df[selection].copy()
df2.columns = df2.columns.str.strip()
df2 = df2[df2['standard_value'].notna() & df2['molecule_chembl_id'].notna() & df2['canonical_smiles'].notna()]

# Calcola il valore pIC50 e sostituisci la colonna standard_value
df2 = norm_value(df2)
df2 = pIC50(df2)

# Salva df2 contenente pIC50 in un file CSV
filename_pic50 = search_term + '_bioactivity_with_pIC50.csv'
df2.to_csv(filename_pic50, index=False)

# Calcola i descrittori di Lipinski
df_lipinski = lipinski(df2.canonical_smiles)

# Resetta gli indici dei DataFrame
df2.reset_index(drop=True, inplace=True)
df_lipinski.reset_index(drop=True, inplace=True)

# Combina i dati originali con i descrittori di Lipinski
df_combined = pd.concat([df2, df_lipinski], axis=1)

# Rimuovi le colonne non necessarie
df_combined = df_combined.drop(columns=['molecule_chembl_id', 'canonical_smiles'], errors='ignore')

# Funzione per calcolare i descrittori aggiuntivi
def additional_descriptors(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        if mol is not None:
            moldata.append(mol)
        else:
            moldata.append(None)

    baseData = []
    for mol in moldata:
        if mol is not None:
            num_atoms = Descriptors.HeavyAtomCount(mol)
            num_bonds = mol.GetNumBonds()
            molar_refractivity = Descriptors.MolMR(mol)
            tpsa = Descriptors.TPSA(mol)
            row = [num_atoms, num_bonds, molar_refractivity, tpsa]
        else:
            row = [np.nan, np.nan, np.nan, np.nan]
        baseData.append(row)

    columnNames = ["NumAtoms", "NumBonds", "MolarRefractivity", "TPSA"]
    descriptors = pd.DataFrame(baseData, columns=columnNames)
    return descriptors

# Calcola i descrittori aggiuntivi
df_additional = additional_descriptors(df2.canonical_smiles)

# Reset degli indici per allineare correttamente i dataframe
df2.reset_index(drop=True, inplace=True)
df_additional.reset_index(drop=True, inplace=True)

# Combina il dataframe originale con i nuovi descrittori
df_combined = pd.concat([df_combined, df_additional], axis=1)

# 8) Creazione delle impronte molecolari
df3 = pd.read_csv(filename_pic50)  # Carica il file contenente pIC50
selection = ['canonical_smiles', 'molecule_chembl_id']
df3_selection = df3[selection]

filename = search_term + '_molecule.smi'
if not os.path.isfile(filename):
    df3_selection.to_csv(filename, sep='\t', index=False, header=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# Ottieni la directory home dell'utente in modo compatibile tra piattaforme
home_dir = os.getenv("HOME") or os.getenv("USERPROFILE")

# Definisci il percorso del file JAR e del file XML
jar_path = os.path.join(home_dir, 'PycharmProjects', 'DrugRepurposing', 'padel', 'PaDEL-Descriptor', 'PaDEL-Descriptor.jar')
xml_path = os.path.join(home_dir, 'PycharmProjects', 'DrugRepurposing', 'padel', 'PaDEL-Descriptor', 'PubchemFingerprinter.xml')

# Comando per eseguire il file JAR con le opzioni specificate
command = [
    'java',
    '-Xms1G',
    '-Xmx1G',
    '-Djava.awt.headless=true',
    '-jar', jar_path,
    '-removesalt',
    '-standardizenitro',
    '-fingerprints',
    '-descriptortypes', xml_path,
    '-dir', './',
    '-file', 'descriptors_output.csv'
]

# Esegui il comando utilizzando Popen
if not os.path.isfile('descriptors_output.csv'):
    print("Esecuzione del calcolo PaDEL dei descrittori in corso, attendere... \n")
    process = subprocess.Popen(command, stdout=PIPE, stderr=PIPE)
    result = process.communicate()
    print(result[0].decode('utf-8'))
    if result[1]:
        print("Errore:", result[1].decode('utf-8'))
else:
    print(f"Esecuzione dello script fallita, in quanto il file 'descriptors_output' esiste già.\n")

# 9) RANDOM FOREST

# Carica i file CSV delle fingerprints
df_fingerprints = pd.read_csv('descriptors_output.csv')

# Droppa le colonne non necessarie
df_fingerprints = df_fingerprints.drop(columns=['Name'], errors='ignore')

# Assicurati che tutti i dati siano numerici
df_fingerprints = df_fingerprints.apply(pd.to_numeric, errors='coerce')
df_combined = df_combined.apply(pd.to_numeric, errors='coerce')

# Combina i descrittori vari con le impronte molecolari
df_combined = pd.concat([df_combined, df_fingerprints], axis=1)

# Gestisci la variabile target (pIC50)
df3_Y = df3['pIC50']

# Combina le impronte molecolari + descrittori con i valori pIC50
dataset = pd.concat([df_combined, df3_Y], axis=1)

# Verifica che il file non esista già prima di salvarlo
filename = search_term + '_bioactivity_preprocessed_data.csv'
if not os.path.isfile(filename):
    dataset.to_csv(filename, index=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# Ricarica il dataset completo
df = pd.read_csv(filename)

# Gestione dei valori NaN e infiniti
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Assicurati che tutti i valori siano numerici e non troppo grandi
df = df[(df < 1e6).all(axis=1)]

# Separazione della variabile target (pIC50) e delle feature
X = df.drop('pIC50', axis=1)
Y = df['pIC50']

print(f"Forma delle feature dopo la pulizia: {X.shape}")

# Analisi della varianza delle feature
variances = X.var()
low_variance_threshold = 0.16
selection = VarianceThreshold(threshold=low_variance_threshold)
X_reduced = selection.fit_transform(X)
print(f"Numero di caratteristiche dopo la rimozione di quelle a bassa varianza: {X_reduced.shape[1]}")

# Split dei dati in training e test set
X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=46)

# Definisci il modello
model = RandomForestRegressor(random_state=46)

# Addestramento del modello
model.fit(X_train, Y_train)

# Predizione
Y_pred = model.predict(X_test)

# Valuta il modello con R^2
r2 = r2_score(Y_test, Y_pred)
print(f"Coefficiente di determinazione R^2: {r2}")

sns.set(color_codes=True)
sns.set_style("white")

# Distribuzione della variabile target pIC50
plt.figure(figsize=(7, 5))
sns.histplot(Y, kde=True, bins=30)
plt.title('Distribuzione di pIC50')
plt.xlabel('pIC50')
plt.ylabel('Frequenza')
plt.show()

ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha': 0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.show()