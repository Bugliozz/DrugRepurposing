# Blocco 2: Ricerca dei Target e Download dei Dati di Bioattività

# Scopo del blocco:
# Questo blocco ha lo scopo di cercare il target biologico di interesse (in questo caso l'acetilcolinesterasi) nel database ChEMBL,
# scaricare i dati di bioattività relativi a questo target e salvarli in un file CSV per analisi future.

# Importa le librerie necessarie
import pandas as pd
from chembl_webresource_client.new_client import new_client
import os

# Imposta il percorso base per salvare i file
base_path = r'C:\Users\marco\PycharmProjects\DrugRepurposing'

# Crea la cartella se non esiste
if not os.path.exists(base_path):
    os.makedirs(base_path)

# Ricerca del target nel database ChEMBL
target = new_client.target
target_query = target.search('acetylcholinesterase')  # Cerca "acetylcholinesterase" nel database
targets = pd.DataFrame.from_dict(target_query)

# Seleziona il target desiderato (assumiamo il primo)
selected_target = targets.iloc[0]
target_chembl_id = selected_target['target_chembl_id']
print(f"Target selezionato: {selected_target['pref_name']} (ChEMBL ID: {target_chembl_id})")

# Download dei dati di bioattività per il target selezionato
activity = new_client.activity
res = activity.filter(target_chembl_id=target_chembl_id).filter(standard_type='IC50')  # Filtra per IC50

# Converti i dati in un DataFrame
df = pd.DataFrame.from_dict(res)

# Salva i dati di bioattività in un file CSV
bioactivity_data_file = os.path.join(base_path, 'bioactivity_data_raw.csv')
df.to_csv(bioactivity_data_file, index=False)
print(f"Dati di bioattività salvati in {bioactivity_data_file}")

# Blocco 3: Pre-elaborazione dei Dati

# Scopo del blocco:
# Questo blocco esegue la pulizia e la preparazione dei dati di bioattività per l'analisi.
# Rimuove duplicati, gestisce valori mancanti, converte dati in formati appropriati e calcola il pIC50.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leggi i dati di bioattività
bioactivity_data_file = os.path.join(base_path, 'bioactivity_data_raw.csv')
df = pd.read_csv(bioactivity_data_file)

# Seleziona le colonne di interesse
df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]

# Rimuovi duplicati basati su 'canonical_smiles'
df = df.drop_duplicates(subset='canonical_smiles')

# Rimuovi record con valori mancanti
df = df.dropna(subset=['canonical_smiles', 'standard_value'])

# Converti 'standard_value' a numerico e gestisci errori
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')

# Rimuovi record con valori non numerici o mancanti in 'standard_value'
df = df.dropna(subset=['standard_value'])

# Filtra i dati per valori di 'standard_value' positivi
df = df[df['standard_value'] > 0]

# Analizza la distribuzione dei valori di 'standard_value' per identificare outlier
plt.figure(figsize=(8, 6))
sns.histplot(df['standard_value'], bins=100)
plt.xlabel('Standard Value (IC50)')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Valori di IC50')
plt.show()

# Calcola il logaritmo dei valori di 'standard_value' per una distribuzione più normale
df['standard_value_log'] = np.log10(df['standard_value'])

# Visualizza la distribuzione dei valori logaritmici
plt.figure(figsize=(8, 6))
sns.histplot(df['standard_value_log'], bins=100)
plt.xlabel('Log10(Standard Value)')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Valori Logaritmici di IC50')
plt.show()

# Identifica e gestisci gli outlier utilizzando lo z-score
from scipy import stats

# Calcola i valori di z-score
df['z_score'] = stats.zscore(df['standard_value_log'])

# Filtra i record con z-score entro 3 deviazioni standard
df = df[(df['z_score'] > -3) & (df['z_score'] < 3)]

# Rimuovi le colonne temporanee
df = df.drop(columns=['standard_value_log', 'z_score'])

# Calcola il pIC50 (trasformazione logaritmica inversa di IC50)
def calculate_pIC50(standard_value):
    return -np.log10(standard_value * 1e-9)

df['pIC50'] = df['standard_value'].apply(calculate_pIC50)

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

# Blocco 4: Calcolo dei Descrittori con RDKit e Analisi Statistica

# Scopo del blocco:
# Calcolare una serie di descrittori molecolari utilizzando RDKit e condurre un'analisi statistica
# per identificare quali descrittori differiscono significativamente tra composti attivi e inattivi.

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import mannwhitneyu

# Leggi i dati pre-elaborati
preprocessed_data_file = os.path.join(base_path, 'bioactivity_data_preprocessed.csv')
df = pd.read_csv(preprocessed_data_file)

# Lista dei descrittori da calcolare con RDKit
descrittori = [
    'MolWt',            # Peso molecolare
    'MolLogP',          # Logaritmo del coefficiente di ripartizione ottanolo/acqua
    'TPSA',             # Superficie polare topologica
    'NumHDonors',       # Numero di donatori di idrogeno
    'NumHAcceptors',    # Numero di accettori di idrogeno
    'NumRotatableBonds',# Numero di legami rotazionali
    'RingCount',        # Numero di anelli
    'NumAromaticRings', # Numero di anelli aromatici
    'FractionCSP3',     # Frazione di atomi sp3 di carbonio
    'BalabanJ',         # Indice di Balaban
    'BertzCT',          # Indice di complessità di Bertz
    'Chi0v',            # Indice Chi di valenza
    'Kappa1',           # Indice kappa 1
    'LabuteASA'         # Superficie accessibile al solvente di Labute
]

# I descrittori sono stati scelti perché forniscono informazioni chiave sulle proprietà fisico-chimiche
# delle molecole che possono influenzare l'attività biologica, come dimensioni, lipofilicità, polarità e struttura.

# Funzione per calcolare i descrittori con RDKit
def calcola_descrittori(df):
    # Aggiungi una colonna 'Mol' con gli oggetti molecola di RDKit
    df['Mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)

    # Rimuovi le molecole non valide (dove 'Mol' è None)
    df = df[df['Mol'].notnull()].reset_index(drop=True)

    # Calcola i descrittori
    for desc in descrittori:
        func = getattr(Descriptors, desc, None)
        if func is not None:
            df[desc] = df['Mol'].apply(func)
        else:
            print(f"Descrittore {desc} non trovato in RDKit.")

    # Rimuovi la colonna 'Mol' poiché non è più necessaria
    df.drop(columns=['Mol'], inplace=True)

    return df

# Calcola i descrittori
df = calcola_descrittori(df)

# Aggiungi una colonna 'Class' basata sul valore di pIC50
df['Class'] = df['pIC50'].apply(lambda x: 'Active' if x > 6 else 'Inactive')

# Livello di significatività
alpha = 0.05

# Lista per memorizzare i risultati
risultati = []

for desc in descrittori:
    # Valori per composti attivi e inattivi
    valori_attivi = df[df['Class'] == 'Active'][desc]
    valori_inattivi = df[df['Class'] == 'Inactive'][desc]

    # Esegui il test statistico di Mann-Whitney U
    stat, p = mannwhitneyu(valori_attivi, valori_inattivi, alternative='two-sided')

    # Determina se c'è una differenza significativa
    if p < alpha:
        significatività = 'Differenza significativa'
    else:
        significatività = 'Nessuna differenza significativa'

    # Salva i risultati
    risultati.append({'Descrittore': desc, 'p-value': p, 'Significatività': significatività})

# Converti i risultati in un DataFrame
risultati_df = pd.DataFrame(risultati)

# Ordina i risultati per p-value crescente
risultati_df = risultati_df.sort_values(by='p-value')

# Stampa la tabella finale dei risultati
print("\nTabella dei risultati ordinati per p-value:")
print(risultati_df)

# Blocco 5: Calcolo dei Descrittori Molecolari e Fingerprint con PaDEL-Descriptor utilizzando padelpy

# Scopo del blocco:
# Utilizzare PaDEL-Descriptor tramite padelpy per calcolare sia i descrittori molecolari 2D che le fingerprint molecolari
# che saranno utilizzati per la modellazione predittiva.
# Le fingerprint forniscono una rappresentazione digitale delle molecole basata sulla presenza o assenza di particolari sub-strutture chimiche.


import os
from padelpy import padeldescriptor
from rdkit import Chem

# Verifica della validità dei SMILES
print("\nVerifica della validità dei SMILES:")
valid_indices = []
invalid_indices = []

for idx, smile in enumerate(df['canonical_smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        valid_indices.append(idx)
    else:
        invalid_indices.append(idx)

print(f"Numero totale di molecole: {len(df)}")
print(f"Numero di molecole valide: {len(valid_indices)}")
print(f"Numero di molecole non valide: {len(invalid_indices)}")

# Rimuovi molecole non valide
if invalid_indices:
    df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"Molecole non valide rimosse: {len(invalid_indices)}")

# Salva gli SMILES validi in un file per PaDEL-Descriptor
smiles_file = os.path.join(base_path, 'molecules_for_padel.smi')
df[['canonical_smiles', 'molecule_chembl_id']].to_csv(
    smiles_file, sep='\t', index=False, header=False
)
print(f"\nFile SMILES creato: {smiles_file}")

# Verifica che il file SMILES sia stato creato correttamente e non sia vuoto
if os.path.exists(smiles_file) and os.path.getsize(smiles_file) > 0:
    print("Il file SMILES è stato creato correttamente ed è pronto per l'elaborazione.")
else:
    print("Errore: il file SMILES non è stato creato correttamente o è vuoto.")
    import sys
    sys.exit("Interruzione dell'esecuzione a causa di un errore nella creazione del file SMILES.")

# Definisci il percorso di output per i descrittori
descriptors_output_file = os.path.join(base_path, 'padel_descriptors_with_fingerprints.csv')
print(f"Percorso di output per i descrittori: {descriptors_output_file}")

# Esegui PaDEL-Descriptor utilizzando padelpy con i parametri corretti
print("\nEsecuzione di PaDEL-Descriptor...")
padeldescriptor(
    mol_dir=smiles_file,
    d_file=descriptors_output_file,
    d_2d=True,            # Calcola i descrittori 2D
    fingerprints=True,    # Calcola le fingerprint molecolari
    retainorder=True,
    removesalt=True,
    detectaromaticity=True,
    standardizenitro=True,
    threads=-1            # Usa tutti i core disponibili
)

# Verifica che il file dei descrittori sia stato creato correttamente e non sia vuoto
if os.path.exists(descriptors_output_file) and os.path.getsize(descriptors_output_file) > 0:
    print("\nDescrittori e fingerprint calcolati con successo!")
    print(f"Descrittori e fingerprint salvati in: {descriptors_output_file}")
else:
    print("\nErrore: il file dei descrittori e delle fingerprint non è stato creato correttamente o è vuoto.")
    import sys
    sys.exit("Interruzione dell'esecuzione a causa di un errore nella creazione del file dei descrittori.")

# Blocco 6: Preparazione dei Dati per la Modellazione

# Scopo del blocco:
# Unire i descrittori calcolati con i valori di pIC50, gestire i valori mancanti e preparare
# la matrice delle caratteristiche (X) e il target (Y) per la modellazione.

import pandas as pd
import numpy as np

# Leggi i descrittori generati da PaDEL
descriptors_output_file = os.path.join(base_path, 'padel_descriptors.csv')
df_descriptors = pd.read_csv(descriptors_output_file)

# Assicurati che gli ID molecolari siano coerenti per il merge
df_descriptors.rename(columns={'Name': 'molecule_chembl_id'}, inplace=True)

# Leggi i dati con i valori di pIC50
preprocessed_data_file = os.path.join(base_path, 'bioactivity_data_preprocessed.csv')
df_pic50 = pd.read_csv(preprocessed_data_file)

# Unisci i dati dei descrittori con i valori di pIC50
df_merged = pd.merge(
    df_descriptors,
    df_pic50[['molecule_chembl_id', 'pIC50']],
    on='molecule_chembl_id',
    how='inner'
)

# Gestione dei valori mancanti
# Sostituisci valori infiniti con NaN
df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

# Rimuovi colonne con più del 50% di valori mancanti
threshold_missing = 0.5
df_merged = df_merged.loc[:, df_merged.isnull().mean() < threshold_missing]

# Prepara X rimuovendo 'molecule_chembl_id' e 'pIC50'
X = df_merged.drop(columns=['molecule_chembl_id', 'pIC50'])

# Seleziona solo colonne numeriche
X = X.select_dtypes(include=[np.number])

# Gestione dei valori mancanti
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Prepara il target Y
Y = df_merged['pIC50'].reset_index(drop=True)

# Rimozione delle caratteristiche con varianza zero
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
selector.fit(X_imputed)
zero_variance_features = X_imputed.columns[~selector.get_support()]
print("Colonne con varianza zero:", zero_variance_features.tolist())

# Rimuovi le colonne con varianza zero
X_reduced = selector.transform(X_imputed)

# Aggiorna i nomi delle colonne dopo la selezione
selected_features = X_imputed.columns[selector.get_support()]

# Converti X_reduced in DataFrame per mantenere i nomi delle colonne
X_reduced = pd.DataFrame(X_reduced, columns=selected_features)

print(f"Dimensione finale della matrice delle caratteristiche: {X_reduced.shape}")

# Blocco 7: Pre-elaborazione con PCA, Ottimizzazione e Modellazione

# Scopo del blocco:
# Eseguire la standardizzazione, ridurre la dimensionalità con PCA, ottimizzare gli iperparametri
# e costruire modelli di regressione e classificazione utilizzando Random Forest.

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, ConfusionMatrixDisplay, r2_score
)
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Standardizzazione delle caratteristiche
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Applicazione della PCA
pca = PCA(n_components=50)  # Numero di componenti principali
X_pca = pca.fit_transform(X_scaled)

# Verifica della varianza spiegata
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"Varianza totale spiegata dalle {pca.n_components_} componenti principali: {explained_variance:.2%}")

# Regressione con Random Forest
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(
    X_pca, Y, test_size=0.2, random_state=200
)

param_grid_reg = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2]
}

grid_search_reg = GridSearchCV(
    RandomForestRegressor(random_state=200),
    param_grid_reg, cv=5, scoring='r2', n_jobs=-1
)
grid_search_reg.fit(X_train_reg, Y_train_reg)

print(f"Migliori Parametri per la Regressione da GridSearch: {grid_search_reg.best_params_}")

best_reg_model = grid_search_reg.best_estimator_

# Predizione e valutazione sul training set
Y_pred_train_reg = best_reg_model.predict(X_train_reg)
r2_train = r2_score(Y_train_reg, Y_pred_train_reg)
print(f"Coefficiente di determinazione R² sul training set: {r2_train:.2f}")

# Predizione e valutazione sul test set
Y_pred_test_reg = best_reg_model.predict(X_test_reg)
r2_test = r2_score(Y_test_reg, Y_pred_test_reg)
print(f"Coefficiente di determinazione R² sul test set: {r2_test:.2f}")

# Visualizzazione dei risultati di regressione
plt.figure(figsize=(7, 5))
sns.histplot(Y, kde=True, bins=30)
plt.title('Distribuzione di pIC50')
plt.xlabel('pIC50')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(5, 5))
ax = sns.regplot(x=Y_test_reg, y=Y_pred_test_reg, scatter_kws={'alpha': 0.4})
ax.set_xlabel('pIC50 Sperimentale (Test Set)', fontsize='large', fontweight='bold')
ax.set_ylabel('pIC50 Predetto', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
plt.title('Regressione: Predicted vs Experimental pIC50 (Test Set)')
plt.show()

# Classificazione con Random Forest
threshold = 6  # Soglia per classificare i composti come attivi o inattivi
Y_class = (Y >= threshold).astype(int)

X_train_cls, X_test_cls, Y_train_cls, Y_test_cls = train_test_split(
    X_pca, Y_class, test_size=0.2, random_state=200
)

param_grid_cls = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2]
}

grid_search_cls = GridSearchCV(
    RandomForestClassifier(random_state=200),
    param_grid_cls, cv=5, scoring='f1', n_jobs=-1
)
grid_search_cls.fit(X_train_cls, Y_train_cls)

print(f"Migliori Parametri per la Classificazione da GridSearch: {grid_search_cls.best_params_}")

best_cls_model = grid_search_cls.best_estimator_

# Predizione e valutazione sul training set
Y_pred_train_cls = best_cls_model.predict(X_train_cls)
conf_matrix_train = confusion_matrix(Y_train_cls, Y_pred_train_cls)
accuracy_train = accuracy_score(Y_train_cls, Y_pred_train_cls)
precision_train = precision_score(Y_train_cls, Y_pred_train_cls)
recall_train = recall_score(Y_train_cls, Y_pred_train_cls)
f1_train = f1_score(Y_train_cls, Y_pred_train_cls)

print("\nMetriche sul Training Set:")
print(f"Confusion Matrix (Training Set):\n{conf_matrix_train}")
print(f"Accuracy: {accuracy_train:.2f}")
print(f"Precision: {precision_train:.2f}")
print(f"Recall: {recall_train:.2f}")
print(f"F1 Score: {f1_train:.2f}")

ConfusionMatrixDisplay(conf_matrix_train).plot(cmap='Blues')
plt.title('Confusion Matrix (Training Set)')
plt.show()

# Predizione e valutazione sul test set
Y_pred_test_cls = best_cls_model.predict(X_test_cls)
conf_matrix_test = confusion_matrix(Y_test_cls, Y_pred_test_cls)
accuracy_test = accuracy_score(Y_test_cls, Y_pred_test_cls)
precision_test = precision_score(Y_test_cls, Y_pred_test_cls)
recall_test = recall_score(Y_test_cls, Y_pred_test_cls)
f1_test = f1_score(Y_test_cls, Y_pred_test_cls)

print("\nMetriche sul Test Set:")
print(f"Confusion Matrix (Test Set):\n{conf_matrix_test}")
print(f"Accuracy: {accuracy_test:.2f}")
print(f"Precision: {precision_test:.2f}")
print(f"Recall: {recall_test:.2f}")
print(f"F1 Score: {f1_test:.2f}")

ConfusionMatrixDisplay(conf_matrix_test).plot(cmap='Greens')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# Visualizzazione delle classi sulle prime due componenti principali
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_class, cmap='viridis', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Analisi delle Componenti Principali')
plt.colorbar(label='Classe')
plt.show()
