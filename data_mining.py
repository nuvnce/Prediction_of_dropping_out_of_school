import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("assets/students_dataset.csv")

# 1. Exploration et visualisation
# Sauvegarder une heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.savefig("assets/correlation_heatmap.png")
plt.close()

# Histogramme des notes
plt.figure(figsize=(8, 6))
sns.histplot(df["Note_Moyenne"], bins=20)
plt.title("Distribution des notes moyennes")
plt.savefig("assets/notes_histogram.png")
plt.close()

# 2. Clustering (K-Means)
scaler = StandardScaler()
numerical_cols = ["Âge", "Note_Moyenne", "Taux_Absentéisme", "Devoirs_Remis", "Temps_Moodle_Heures", "Participation_Forums", "Satisfaction"]
X_cluster = scaler.fit_transform(df[numerical_cols])
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_cluster)
df.to_csv("assets/students_with_clusters.csv", index=False)

# Visualisation des clusters (Âge vs Note_Moyenne)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Âge", y="Note_Moyenne", hue="Cluster", data=df)
plt.title("Clusters d’étudiants (Âge vs Notes)")
plt.savefig("assets/clusters.png")
plt.close()

# 3. Classification supervisée (Random Forest)
le = LabelEncoder()
categorical_cols = ["Sexe", "Région", "Niveau_Éducation_Parents", "Abandon"]
for col in categorical_cols:
    df[col + "_encoded"] = le.fit_transform(df[col])

features = ["Âge", "Sexe_encoded", "Région_encoded", "Niveau_Éducation_Parents_encoded",
            "Note_Moyenne", "Taux_Absentéisme", "Devoirs_Remis", "Temps_Moodle_Heures",
            "Participation_Forums", "Satisfaction"]
X = df[features]
y = df["Abandon_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision Random Forest : {accuracy:.2f}")

# Importance des variables
importances = pd.DataFrame({"Feature": features, "Importance": clf.feature_importances_})
importances = importances.sort_values("Importance", ascending=False)
importances.to_csv("assets/feature_importance.csv", index=False)

# 4. Règles d’association
binary_df = pd.DataFrame()
binary_df["Faibles_Notes"] = (df["Note_Moyenne"] < 10).astype(int)
binary_df["Haut_Absentéisme"] = (df["Taux_Absentéisme"] > 30).astype(int)
binary_df["Faible_Engagement"] = (df["Temps_Moodle_Heures"] < 10).astype(int)
binary_df["Abandon"] = (df["Abandon"] == "Oui").astype(int)

frequent_itemsets = apriori(binary_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
rules.to_csv("assets/association_rules.csv", index=False)
print("Règles d’association générées : association_rules.csv")