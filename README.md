# Prévention de l’Abandon Scolaire grâce au Data Mining

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit)](YOUR_STREAMLIT_LINK)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Objectif du Projet

Utiliser des **techniques de data mining** (classification, clustering, règles d’association) pour :
- Identifier les **facteurs prédictifs** de l’abandon scolaire dans une université.
- Proposer un **outil interactif** permettant de prédire les **risques d’abandon** d’un étudiant donné.

---

## Données Utilisées

**Dataset simulé** (500 étudiants) comprenant :
- **Informations socio-démographiques** : âge, sexe, région, niveau scolaire des parents
- **Résultats académiques** : notes, taux d’absentéisme, remise des devoirs
- **Engagement** : temps passé sur Moodle, participation aux forums
- **Satisfaction** : enquête de feedback
- **Variable cible** : `Abandon` (Oui/Non)


---

## Techniques de Data Mining Appliquées

| Étape | Technique | Objectif |
|------|----------|--------|
| 1 | **Exploration & Visualisation** | Profils types, corrélations, histogrammes |
| 2 | **Clustering (K-Means)** | Identifier des groupes d’étudiants similaires |
| 3 | **Classification (Random Forest)** | Prédire le risque d’abandon |
| 4 | **Règles d’association (Apriori)** | Découvrir des schémas comportementaux |
| 5 | **Sélection de variables** | Importance des features (interprétabilité) |

---

## Application Streamlit – Fonctionnalités

### 1. **Dashboard Exploratoire**
- Heatmaps, histogrammes, visualisations des clusters
- Statistiques globales : taux d’abandon, facteurs clés

### 2. **Simulation Individuelle**
- Formulaire interactif (sliders/dropdowns)
- Prédiction en temps réel du **risque d’abandon** (%)
- Attribution du **profil (cluster)**

### 3. **Recommandations Personnalisées**
- Basées sur les règles d’association
- Ex : *"Suivi personnalisé"*, *"Encourager la participation aux forums"*

### 4. **Téléchargement de Rapports**
- **PDF** : Rapport détaillé par étudiant (caractéristiques, score, recommandations)
- **CSV** : Historique complet des prédictions

---

## Structure du Projet

```
Abandon/
│
├── app.py                     → Application Streamlit principale
├── data_mining.py             → Analyse, clustering, classification, règles
├── generate_dataset.py        → Génération du dataset simulé
├── requirements.txt           → Dépendances Python
├── .gitignore                 
├── README.md                  

```

---

## Installation & Lancement Local

```bash
# 1. Cloner le repo
git clone https://github.com/ton-user/Abandon.git
cd Abandon

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Générer les données et modèles
python generate_dataset.py
python data_mining.py

# 5. Lancer l'application
streamlit run app.py
```

> L’app s’ouvre automatiquement dans votre navigateur : `http://localhost:8501`


---
## Aperçu de l'Interface

![Interface App1](Assets/Dashboard.png)

![Interface App2](Assets/Prédiction.png)
*Capture d'écran de l'application Streamlit en action*

> Description : Dashboard avec visualisations interactives et formulaire de prédiction.
---

## Auteur

**Daniel ESSONANI**  
*Étudiant en Big Data / Intelligence Artificielle*  
GitHub: [github.com/nuvnce](https://github.com/nuvnce)

---

## Licence

[MIT License](LICENSE) – Libre d’utilisation et de modification.

---

> **Projet académique – Juin 2025**
