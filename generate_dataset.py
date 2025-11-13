import pandas as pd
import numpy as np
import random

# Configurer la seed pour reproductibilité
np.random.seed(42)
random.seed(42)

# Listes de valeurs possibles
sexes = ["Homme", "Femme"]
regions = ["Urbain", "Rural", "Périurbain"]
parent_education = ["Aucun", "Secondaire", "Supérieur"]
abandon_labels = ["Oui", "Non"]

# Générer 500 étudiants
n_students = 500
data = {
    "ID_Étudiant": [f"Etudiant_{i}" for i in range(1, n_students + 1)],
    "Âge": [random.randint(18, 30) for _ in range(n_students)],
    "Sexe": [random.choice(sexes) for _ in range(n_students)],
    "Région": [random.choice(regions) for _ in range(n_students)],
    "Niveau_Éducation_Parents": [random.choice(parent_education) for _ in range(n_students)],
    "Note_Moyenne": [round(random.uniform(0, 20), 1) for _ in range(n_students)],
    "Taux_Absentéisme": [random.randint(0, 50) for _ in range(n_students)],
    "Devoirs_Remis": [random.randint(0, 100) for _ in range(n_students)],
    "Temps_Moodle_Heures": [random.randint(0, 50) for _ in range(n_students)],
    "Participation_Forums": [random.randint(0, 20) for _ in range(n_students)],
    "Satisfaction": [random.randint(1, 10) for _ in range(n_students)],
    "Abandon": [random.choice(abandon_labels) for _ in range(n_students)]
}

# Créer le DataFrame
df = pd.DataFrame(data)

# Sauvegarder en CSV
df.to_csv("assets/students_dataset.csv", index=False)
print("Dataset généré : students_dataset.csv")