import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Housing.csv')
print(df.columns)
# 3. Inspection rapide
print("--- INFO SUR LES DONNÉES ---")
print(df.info()) 
print("\n--- STATISTIQUES DESCRIPTIVES ---")
print(df.describe())


# 1. Histogramme : La répartition des prix
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=30, kde=True, color='teal')
plt.title('Répartition des Prix de Vente')
plt.xlabel('Prix')
plt.ylabel('Nombre de biens')
plt.show()

# 2. Nuage de points : Surface vs Prix (coloré par la Climatisation)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='area', y='price', hue='airconditioning', data=df)
plt.title('Prix en fonction de la surface et de la climatisation')
plt.show()

# 3. Boîte à moustaches (Boxplot) : Prix selon l'ameublement
plt.figure(figsize=(8, 5))
sns.boxplot(x='furnishingstatus', y='price', data=df)
plt.title('Prix selon le niveau d\'ameublement')
plt.show()

# 1. On utilise la fonction magique de Pandas pour tout encoder
# Le paramètre dtype=int force Pandas à mettre des 1 et des 0 (au lieu de True/False)
df_numerique = pd.get_dummies(df, drop_first=True, dtype=int)

# 2. On affiche les nouvelles colonnes pour voir le résultat
print("--- NOUVELLES COLONNES ---")
print(df_numerique.columns)

# 3. On regarde les 5 premières lignes du nouveau tableau
print("\n--- APERÇU DES DONNÉES ENCODÉES ---")
print(df_numerique.head())



# 1. On sépare ce qu'on veut prédire (y) des caractéristiques (X)
# X = Tout notre tableau SAUF la colonne des prix (axis=1 signifie "colonne")
X = df_numerique.drop('price', axis=1)

# y = UNIQUEMENT la colonne des prix
y = df_numerique['price']

# 2. On coupe tout en deux au hasard (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. On affiche la taille de nos nouveaux groupes
print("Taille des données d'entraînement (X_train) :", X_train.shape)
print("Taille des données de test (X_test) :", X_test.shape)

modele_lineaire = LinearRegression()

# 2. ENTRAÎNEMENT (Il apprend sur les 80% de données Train)
modele_lineaire.fit(X_train, y_train)

# 3. ÉVALUATION (Il passe l'examen sur les 20% de données Test cachées)
score_r2 = modele_lineaire.score(X_test, y_test)

# 4. On affiche le résultat de l'examen !
print(f"Le score R² de notre modèle est de : {score_r2 * 100:.2f} %")