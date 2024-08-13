import os
import subprocess
from subprocess import PIPE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chembl_webresource_client.new_client import new_client
from rdkit import Chem  # rdkit ci permette di computare i descrittori molecolari dei composti nei dataset che vengono compilati
from rdkit.Chem import Descriptors, Lipinski
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Configura Pandas per visualizzare tutte le colonne e le righe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("\n")

# 1) RICERCA DEI TARGET

# I target mostrati, si riferiscono a proteine, enzimi, o altri componenti biologici della patologia (o ad essa associati)
# che possono essere presi di mira dai farmaci per trattare o interferire con con essa.
# Più nello specifico, un target (o bersaglio molecolare) in biologia molecolare e farmacologia è una molecola,
# di solito una proteina come un enzima, un recettore, o un canale ionico, che è coinvolta in un processo biologico specifico
# e che può essere influenzata da un farmaco o da un altro tipo di agente bioattivo.

# cross_references: riferimenti incrociati a database esterni o altri identificatori rilevanti per il target.
# organism: indica l'organismo in cui il target è stato identificato.
# pref_name: nome preferito o comune del target.
# score: valore numerico che riflette la rilevanza o la confidenza del target. Valori più alti indicano una maggiore rilevanza.
# species_group_flag: indica se il target appartiene a un gruppo di specie.
# target_chembl_id: identificatore unico del target nel database ChEMBL.
# target_components: elenca i componenti del target. Può contenere informazioni dettagliate su specifiche proteine o parti del target.
# target_type: Indica se il target è un organismo intero, una proteina singola, o un altro tipo di entità.
# tax_id: identificatore tassonomico dell'organismo, un codice numerico che identifica in modo univoco la specie associata al target.

search_term = 'acetylcholinesterase'
target = new_client.target  # Crea un'istanza del client per il target
target_query = target.search(search_term)  # Cerca i target
targets = pd.DataFrame.from_dict(target_query)  # Converte i risultati della ricerca in un DataFrame
print("PER LA RICERCA DEI TARGET DI " + search_term.upper() + " SONO STATI TROVATI I SEGUENTI RISULTATI: \n" + targets.to_string() + "\n")  # Stampa il DataFrame per visualizzare tutte le colonne

# 2) SELEZIONARE E RECUPERARE I DATI DI BIOATTIVITÀ

# In questa fase, dopo aver selezionato il target, otterniamo i dati di bioattività relativi a composti chimici testati contro il target
# questi dati sono specifici per la misura IC50 (Inhibitory Concentration 50), che rappresenta la concentrazione necessaria di un composto per inibire il 50% dell'attività biologica della proteasi target.

numero = int(input("Inserisci il numero del target: "))
selected_target = targets.target_chembl_id[numero]  # Seleziona il settimo target (indice 6)
print("\nL'ID SELEZIONATO è: " + selected_target + "\n")  # Stampa l'ID del target selezionato
activity = new_client.activity  # Crea un'istanza del client per l'attività
res = activity.filter(target_chembl_id=selected_target).filter(
    standard_type="IC50")  # Filtra le attività per il target selezionato e per il tipo standard "IC50" -> parametro utilizzato per valutare l'efficacia di una sostanza nell'inibire il target
df = pd.DataFrame.from_dict(res)  # Converte i risultati della ricerca in un DataFrame
#print("LE MOLECOLE BIOATTIVE (comprese di standardValue = 0) CON LA PROTEINA SELEZIONATA SONO: \n " + df.to_string() + "\n")  # Stampa del DataFrame con tutte le entry (comprese quelle con standardValue=0)
filename = search_term + '_bioactivity_data.csv'
if not os.path.isfile(filename):  # Esporta il DataFrame in un file CSV solo se il file non esiste
    df.to_csv(filename, index=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# 3) PRE-ELABORAZIONE DEI DATI SULLA BIOATTIVITÀ E CREAZIONE DEI FILE CSV

# I dati di bioattività sono nell'unità IC50.
# I composti con valori inferiori a 1000 nM saranno considerati attivi
# Mentre quelli superiori a 10.000 nM saranno considerati inattivi.
# I valori compresi tra 1.000 e 10.000 nM saranno considerati intermedi.

selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']  # SMILES (acronimo di Simplified Molecular Input Line Entry System[1]) è un metodo per descrivere la struttura di una molecola usando una breve stringa ASCII.
df2 = df[selection].copy()
df2.columns = df2.columns.str.strip()
df2 = df2[df2['standard_value'].notna() & df2['molecule_chembl_id'].notna() & df2['canonical_smiles'].notna()]
bioactivity_class = []
for i in df2.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append("inactive")
    elif float(i) <= 1000:
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")
df2['bioactivity_class'] = bioactivity_class

# 4) CARICARE I DATI SULLA BIOATTIVITA' E CALCOLARE I DESCRITTORI DI LIPINSKI

# Utilizzando la libreria rdkit, andremo ad analizzare la struttura chimica dei composti bioattivi mediante la loro notazione S.M.I.L.E.S
# con il fine di calcolare i descrittori molecolari, utili a verificare le regole di Lipinski

# Christopher Lipinski, uno scienziato della Pfizer, ha ideato un insieme di regole pratiche per valutare la "druglikeness" dei composti,
# ovvero la loro somiglianza a farmaci potenziali.
# Questa "druglikeness" si basa sull'assorbimento, distribuzione, metabolismo ed escrezione (ADME), noto anche come profilo farmacocinetico.
# Lipinski ha analizzato tutti i farmaci approvati dalla FDA che sono attivi per via orale, formulando quelle che sono conosciute come Regole di Lipinski.
# Le Regole di Lipinski affermano quanto segue:
#   - Peso molecolare < 500 Dalton
#   - Coefficiente di ripartizione ottanolo-acqua (LogP) < 5
#   - Donatori di legami idrogeno < 5
#   - Accettori di legami idrogeno < 10

def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        if mol is not None:  # Aggiungi solo molecole valide
            moldata.append(mol)
        else:
            moldata.append(None)  # Inserisci un segnaposto per mantenere l'allineamento

    baseData = []
    for mol in moldata:
        if mol is not None:
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

            row = [desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]
        else:
            row = [np.nan, np.nan, np.nan, np.nan]  # Inserisci NaN per molecole non valide

        baseData.append(row)

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(baseData, columns=columnNames)
    return descriptors

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    return descriptors


df_lipinski = lipinski(df2.canonical_smiles)
df2.reset_index(drop=True, inplace=True)
df_lipinski.reset_index(drop=True, inplace=True)
df_combined = pd.concat([df2, df_lipinski], axis=1)

#print("ELENCO DELLE PRECEDENTI MOLECOLE CON I RELATIVI VALORI UTILI ALLE REGOLE DI LIPINSKI: \n ")
#print(df_combined.to_string() + "\n")

# 5) CONVERSIONE DA IC50 A pIC50

# Per ottenere una distribuzione più uniforme dei dati di IC50, convertiremo i valori di IC50 nella scala logaritmica negativa, utilizzando -log10(IC50).
# Questa funzione personalizzata, denominata pIC50(), accetterà un DataFrame come input e farà quanto segue:
#  -Prenderà i valori IC50 dalla colonna standard_value e li convertirà da nM a M moltiplicando il valore per 10.
#  -Prenderà il valore molare e applicherà -log10.
#  -Eliminerà la colonna standard_value e creerà una nuova colonna denominata pIC50.

def pIC50(input, threshold_active=6, threshold_inactive=5):
    pIC50 = []
    min_value = 1e-10  # Valore minimo per evitare il log di zero

    for i in input['standard_value_norm']:
        i = float(i)
        if i == 0:
            i = min_value  # Sostituisci zero con il valore minimo
        molar = i * (10 ** -9)  # Converte nM in M
        pIC50_value = -np.log10(molar)
        pIC50.append(pIC50_value)

    input['pIC50'] = pIC50

    # Classifica in base a pIC50
    bioactivity_class = []
    for value in input['pIC50']:
        if value >= threshold_active:
            bioactivity_class.append("active")
        elif value < threshold_inactive:
            bioactivity_class.append("inactive")
        else:
            bioactivity_class.append("intermediate")

    input['bioactivity_class'] = bioactivity_class

    # Rimuovi la colonna standard_value_norm (se non più necessaria)
    x = input.drop('standard_value_norm', axis=1)

    return x


# Nota: i valori superiori a 100.000.000 verranno fissati su 100.000.000 altrimenti il valore logaritmico negativo diventerà negativo.

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        i = float(i)
        if i > 100000000:
            i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', axis=1)

    return x


# Ora stampiamo il dataset con i valori IC50 trasformati in pIC50 normalizzati
df_norm = norm_value(df_combined)
df_final = pIC50(df_norm)
print("ELENCO DELLE PRECEDENTI MOLECOLE + LIPINSKI DESCRIPTOR E IC50 TRASFORMATO IN pIC50 NORMALIZZATO: \n " + df_final.head(100).to_string() + "\n")

filename = search_term + '_bioactivity_preprocessed_data.csv'
if not os.path.isfile(filename):  # Esporta il DataFrame in un file CSV solo se il file non esiste
    df_final.to_csv(filename, index=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# 6) DATA ANALISI ESPLORATIVA (ANALISI DELLO SPAZIO CHIMICO) MEDIANTE I DESCRITTORI DI LIPINSKI

# Andremo a utilizzare un test non parametrico, ossia il Mann-Whitney U Test
# per confrontare la distribuzione dei descrittori di Lipinski nei due gruppi indipendenti (composti attivi vs inattivi).

# Lo scopo del test Mann-Whitney U è utile a verificare se le medie dei ranghi dei due gruppi sono significativamente diverse,
# indicando che uno dei gruppi tende ad avere valori più alti o più bassi rispetto all'altro.
# Lo utilizzeremo per confrontare due gruppi in termini di un singolo descrittore (ad esempio, la massa molecolare)
# per verificare se c'è una differenza significativa tra i due gruppi, quindi rispondere a domande del tipo:
# "la massa molecolare media dei composti attivi è significativamente diversa da quella dei composti inattivi?"
# Ad esempio, se i composti attivi tendono ad avere un peso molecolare significativamente più basso rispetto a quelli inattivi,
# ciò potrebbe indicare che il peso molecolare è un fattore critico per l'attività nel contesto di quel target.
# L'ipotesi nulla (H0) nel test di Mann-Whitney afferma che la distribuzione del descrittore per i composti attivi è la stessa della distribuzione del descrittore per i composti inattivi.
# L'ipotesi alternativa (H1) suggerisce che c'è una differenza sistematica tra queste distribuzioni, ovvero che il descrittore tende ad avere valori più alti (o più bassi) in uno dei due gruppi.

df_final_no_intermediate = df_final[df_final.bioactivity_class != 'intermediate']  # Eliminiamo i record dei composti con bioattività intermedia


def mannwhitney(descriptors, verbose=False):
    from numpy.random import seed
    from scipy.stats import mannwhitneyu

    seed(1)  # Seed the random number generator

    # Lista per raccogliere i risultati per tutti i descrittori
    all_results = []

    # Itera su ciascun descrittore
    for descriptor in descriptors:
        # Seleziona le colonne di interesse
        selection = [descriptor, 'bioactivity_class']
        df_MW = df_final_no_intermediate[selection]

        # Separa i composti attivi e inattivi
        active = df_MW[df_MW.bioactivity_class == 'active'][descriptor]
        inactive = df_MW[df_MW.bioactivity_class == 'inactive'][descriptor]

        # Confronta i campioni
        stat, p = mannwhitneyu(active, inactive)

        # Interpreta i risultati
        alpha = 0.05
        if p > alpha:
            interpretation = 'Same distribution (fail to reject H0)'
            conclusion = f"There is not enough evidence to conclude that the distribution of {descriptor} differs significantly between active and inactive compounds."
        else:
            interpretation = 'Different distribution (reject H0)'
            conclusion = f"There is significant evidence to suggest that the distribution of {descriptor} differs between active and inactive compounds."

        # Memorizza i risultati in un DataFrame temporaneo
        result = pd.DataFrame({
            'Descriptor': [descriptor],
            'Statistics': [stat],
            'p-value': [p],
            'alpha': [alpha],
            'Interpretation': [interpretation],
            'Conclusion': [conclusion]
        })

        # Aggiungi il risultato alla lista complessiva
        all_results.append(result)

    # Combina tutti i risultati in un unico DataFrame
    final_results = pd.concat(all_results, ignore_index=True)

    # Stampa tutti i risultati in una tabella
    print("Test di Mann-Whitney U: \n" + final_results.to_string(index=False) + "\n")

    # Salva il risultato complessivo in un file CSV
    filename = search_term + '_MannWhitneyU_results.csv'
    if not os.path.isfile(filename):
        final_results.to_csv(filename, index=False)
        print(f"File '{filename}' creato con successo.\n")
    else:
        print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

    return final_results


# Prima di tutto stampiamo un grafico plot box per mostare la relativa frequenza dei composti attivi e inattivi
sns.set(style='ticks')
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='bioactivity_class', data=df_final_no_intermediate, edgecolor='black')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x='bioactivity_class', y='pIC50', data=df_final_no_intermediate)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x='bioactivity_class', y='MW', data=df_final_no_intermediate)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x='bioactivity_class', y='LogP', data=df_final_no_intermediate)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x='bioactivity_class', y='NumHDonors', data=df_final_no_intermediate)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x='bioactivity_class', y='NumHAcceptors', data=df_final_no_intermediate)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')
plt.show()

descriptors = ['pIC50', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
results = mannwhitney(descriptors)

# 7) descrittori aggiuntivi

# Funzione per calcolare i descrittori aggiuntivi
def additional_descriptors(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        if mol is not None:  # Aggiungi solo molecole valide
            moldata.append(mol)
        else:
            moldata.append(None)  # Inserisci un segnaposto per mantenere l'allineamento

    baseData = []
    for mol in moldata:
        if mol is not None:
            # Calcolo dei descrittori aggiuntivi
            num_atoms = Descriptors.HeavyAtomCount(mol)  # Numero di atomi pesanti
            num_bonds = mol.GetNumBonds()  # Numero di legami
            molar_refractivity = Descriptors.MolMR(mol)  # Refrattività molare
            tpsa = Descriptors.TPSA(mol)  # Superficie polare topologica

            row = [num_atoms, num_bonds, molar_refractivity, tpsa]
        else:
            row = [np.nan, np.nan, np.nan, np.nan]  # Inserisci NaN per molecole non valide

        baseData.append(row)

    columnNames = ["NumAtoms", "NumBonds", "MolarRefractivity", "TPSA"]
    descriptors = pd.DataFrame(baseData, columns=columnNames)
    return descriptors


# Carica il file CSV iniziale contenente i dati delle molecole
df2 = pd.read_csv(search_term + '_bioactivity_preprocessed_data.csv')

# Calcola i descrittori aggiuntivi
df_additional = additional_descriptors(df2.canonical_smiles)

# Reset degli indici per allineare correttamente i dataframe
df2.reset_index(drop=True, inplace=True)
df_additional.reset_index(drop=True, inplace=True)

# Combina il dataframe originale con i nuovi descrittori
df_combined2 = pd.concat([df2, df_additional], axis=1)

# Visualizza i primi record del dataframe combinato
print(df_combined2.head())

# Salva il DataFrame combinato in un file CSV
filename = search_term + '_descriptors_with_additional_features.csv'
if not os.path.isfile(filename):  # Esporta il DataFrame in un file CSV solo se il file non esiste
    df_combined2.to_csv(filename, index=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# 8) Creazione delle impronte molecolari

df3 = pd.read_csv(filename)
selection = ['canonical_smiles', 'molecule_chembl_id']
df3_selection = df3[selection]

filename = search_term + '_molecule.smi'
if not os.path.isfile(filename):  # Esporta il DataFrame in un file CSV solo se il file non esiste
    df3_selection.to_csv(filename, sep='\t', index=False, header=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# Ottieni la directory home dell'utente in modo compatibile tra piattaforme
home_dir = os.getenv("HOME") or os.getenv("USERPROFILE")

# Definisci il percorso del file JAR e del file XML
jar_path = os.path.join(home_dir, 'PycharmProjects', 'DrugsRepositioning', 'padel', 'PaDEL-Descriptor', 'PaDEL-Descriptor.jar')
xml_path = os.path.join(home_dir, 'PycharmProjects', 'DrugsRepositioning', 'padel', 'PaDEL-Descriptor', 'PubchemFingerprinter.xml')

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

# Carica i file CSV iniziali
df_fingerprints = pd.read_csv('descriptors_output.csv')
df_lipinski = pd.read_csv(search_term + '_bioactivity_preprocessed_data.csv')
df_additional = pd.read_csv(search_term + '_descriptors_with_additional_features.csv')

# Droppa le colonne non necessarie
df_fingerprints = df_fingerprints.drop(columns=['Name'], errors='ignore')
df_additional = df_additional.drop(columns=['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'pIC50'])

# Assicurati che tutti i dati siano numerici
df_fingerprints = df_fingerprints.apply(pd.to_numeric, errors='coerce')
df_additional = df_additional.apply(pd.to_numeric, errors='coerce')

# Combina i descrittori di Lipinski con le impronte molecolari
df_combined = pd.concat([df_fingerprints, df_additional], axis=1)

# Gestisci la variabile target (pIC50)
df3_Y = df3['pIC50']

# Combina le impronte molecolari + descrittori di Lipinski con i valori pIC50
dataset = pd.concat([df_combined, df3_Y], axis=1)

# Verifica che il file non esista già prima di salvarlo
filename = search_term + 'pubchem_fp.csv'
if not os.path.isfile(filename):
    dataset.to_csv(filename, index=False)
    print(f"File '{filename}' creato con successo.\n")
else:
    print(f"Creazione del file '{filename}' abortita in quanto esiste già.\n")

# Ricarica il dataset completo
df = pd.read_csv(filename)

# Gestione dei valori NaN e infiniti
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Sostituisce inf e -inf con NaN
df.dropna(inplace=True)  # Rimuove tutti i valori NaN

# Assicurati che tutti i valori siano numerici e non troppo grandi
df = df[(df < 1e6).all(axis=1)]  # Filtra valori estremamente grandi

# Separazione della variabile target (pIC50) e delle feature
X = df.drop('pIC50', axis=1)
Y = df['pIC50']

print(f"Forma delle feature dopo la pulizia: {X.shape}")

# Analisi della varianza delle feature
variances = X.var()
low_variance_threshold = 0.1  # Cambia questo valore in base all'analisi
selection = VarianceThreshold(threshold=low_variance_threshold)
X_reduced = selection.fit_transform(X)
print(f"Numero di caratteristiche dopo la rimozione di quelle a bassa varianza: {X_reduced.shape[1]}")

# Split dei dati in training e test set
X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=200)

# Definisci il modello
model = RandomForestRegressor(random_state=200)

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


#bubb