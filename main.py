import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ton_fichier.csv')
print(df.columns)
# 3. Inspection rapide
print("--- INFO SUR LES DONNÉES ---")
print(df.info()) 
print("\n--- STATISTIQUES DESCRIPTIVES ---")
print(df.describe())

# 4. Visualisation 1 : L'histogramme des loyers
#plt.figure(figsize=(8, 5))
#sns.histplot(df['Loyer_Euros'], bins=30, kde=True, color='blue')
#plt.title('Répartition des Loyers')
#plt.xlabel('Loyer en Euros')
#plt.ylabel('Nombre d\'appartements')
#plt.show()

# 5. Visualisation 2 : Le nuage de points (Surface vs Loyer)
#plt.figure(figsize=(8, 5))
#sns.scatterplot(x='Surface_m2', y='Loyer_Euros', hue='Localisation', data=df)
#plt.title('Loyer en fonction de la surface et de la localisation')
#plt.show()