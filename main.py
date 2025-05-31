import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Charger les données
df = pd.read_csv('Financial_inclusion_dataset.csv')
print(df.head())

# 2. Nettoyage
df = df.drop_duplicates()

# Remplir les valeurs manquantes
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encoder les colonnes catégorielles
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    print(col, ":", df[col].unique())
    df[col] = le.fit_transform(df[col])

# 3. Séparer les features et la cible
X = df.drop(['bank_account', 'uniqueid'], axis=1)
y = df['bank_account']

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modèle
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Évaluation
y_pred = clf.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Sauvegarder le modèle
joblib.dump(clf, 'model.pkl')
print("Modèle sauvegardé sous model.pkl ✅")

# 8. Affichage de l'importance des variables
importances = clf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Importance des variables")
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()
