import streamlit as st
import pandas as pd
import plotly.express as px
from data_mining import clf, le, features, kmeans, scaler, numerical_cols
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from io import BytesIO

import os


# === CONFIG ===
st.set_page_config(page_title="Prévention Abandon Scolaire", page_icon="graduation cap", layout="wide")

# === CSS ===
st.markdown("""
<style>
    .main {background: linear-gradient(to bottom, #f0f4f8, #d9e2ec);}
    .header {background: linear-gradient(90deg, #2196F3, #21CBF3); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;}
    .kpi {font-size: 2rem; font-weight: bold; text-align: center; padding: 1rem; border-radius: 12px; margin: 0.5rem;}
    .low {background-color: #e6f4ea; color: #2e7d32;}
    .medium {background-color: #fff3e0; color: #ef6c00;}
    .high {background-color: #ffebee; color: #c62828;}
    .stButton>button {background: #2196F3; color: white; border-radius: 10px; padding: 0.6rem 1.2rem;}
    .stButton>button:hover {background: #1976D2;}
    .stExpander {border-radius: 12px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# === DATA CACHE ===
@st.cache_data
def load_data():
    return pd.read_csv("assets/students_with_clusters.csv")

@st.cache_data
def load_rules():
    return pd.read_csv("assets/association_rules.csv")

@st.cache_data
def load_importance():
    return pd.read_csv("assets/feature_importance.csv")

# === PDF GENERATOR ===
def generate_pdf(data, risk, cluster, recs):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFillColor(colors.HexColor("#2196F3"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 780, "Rapport de Risque d'Abandon")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    y = 740
    for k, v in data.items():
        if "encoded" not in k:
            c.drawString(70, y, f"{k}: {v}")
            y -= 20
    c.drawString(70, y-20, f"Risque: {risk:.1%}")
    c.drawString(70, y-40, f"Cluster: {cluster}")
    c.drawString(70, y-70, "Recommandations:")
    for r in recs[:3]:
        c.drawString(90, y-90-(recs.index(r)*15), f"- {r}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# === SIDEBAR NAV ===
pages = ["Accueil", "Dashboard", "Prédiction", "Recommandations", "Historique", "Rapport"]
page = st.sidebar.radio("Navigation", pages, format_func=lambda x: f"{x}")

# === PAGES ===
if page == "Accueil":
    st.markdown("<div class='header'><h1>Prévention de l'Abandon Scolaire</h1><p>Outil IA pour détecter et agir tôt</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    df = load_data()
    dropout_rate = df['Abandon'].value_counts(normalize=True).get('Oui', 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='kpi low'>{(1-dropout_rate):.1%}<br><small>Réussite</small></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi medium'>{dropout_rate:.1%}<br><small>Abandon</small></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi high'>{len(df)}<br><small>Étudiants</small></div>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/100/000000/graduation-cap.png", width=100)

elif page == "Dashboard":
    st.header("Tableau de bord")
    df = load_data()
    col1, col2 = st.columns([3,2])
    with col1:
        fig = px.scatter_3d(df, x="Âge", y="Note_Moyenne", z="Temps_Moodle_Heures", color="Abandon",
                            size="Taux_Absentéisme", title="Profil 3D des étudiants")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.image("assets/correlation_heatmap.png", caption="Corrélations")
        st.dataframe(load_importance().head(5).style.format({"Importance": "{:.3f}"}))

elif page == "Prédiction":
    st.header("Prédire le risque")
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            id_std = st.text_input("ID Étudiant", "E001")
            age = st.slider("Âge", 18, 30, 22)
            sex = st.selectbox("Sexe", ["Homme", "Femme"])
            region = st.selectbox("Région", ["Urbain", "Rural", "Périurbain"])
        with col2:
            grade = st.slider("Note moyenne", 0.0, 20.0, 12.0, 0.1)
            absence = st.slider("Absentéisme (%)", 0, 50, 15)
            homework = st.slider("Devoirs remis (%)", 0, 100, 70)
            moodle = st.slider("Temps Moodle (h)", 0, 50, 20)
        submitted = st.form_submit_button("Prédire")
    if submitted:
        with st.spinner("Analyse en cours..."):
            student = {"ID_Étudiant": id_std, "Âge": age, "Sexe": sex, "Région": region,
                       "Niveau_Éducation_Parents": "Secondaire", "Note_Moyenne": grade,
                       "Taux_Absentéisme": absence, "Devoirs_Remis": homework,
                       "Temps_Moodle_Heures": moodle, "Participation_Forums": 5, "Satisfaction": 7}
            df_std = pd.DataFrame([student])
            for c in ["Sexe", "Région", "Niveau_Éducation_Parents"]:
                df_std[c + "_encoded"] = le.fit_transform(df_std[c])
            X = df_std[features]
            risk = clf.predict_proba(X)[0][1]
            cluster = kmeans.predict(scaler.transform(df_std[numerical_cols]))[0]
            rules = load_rules()
            recs = []
            for _, r in rules.iterrows():
                if "Abandon" in r["consequents"]:
                    ants = eval(r["antecedents"])
                    if (grade < 10 and "Faibles_Notes" in ants) or (absence > 30 and "Haut_Absentéisme" in ants):
                        recs.append(f"{ants} → Suivi personnalisé")
            # Save
            if "hist" not in st.session_state: st.session_state.hist = pd.DataFrame()
            df_std["Risque"] = risk; df_std["Cluster"] = cluster
            st.session_state.hist = pd.concat([st.session_state.hist, df_std], ignore_index=True)
            st.session_state.hist.to_csv("assets/student_history.csv", index=False)
        # Results
        st.subheader("Résultat")
        if risk < 0.3:
            st.markdown(f"<div class='low'>Risque faible : {risk:.1%}</div>", unsafe_allow_html=True)
        elif risk < 0.7:
            st.markdown(f"<div class='medium'>Risque modéré : {risk:.1%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='high'>Risque élevé : {risk:.1%}</div>", unsafe_allow_html=True)
        st.write(f"**Cluster** : {cluster}")
        if recs:
            with st.expander("Recommandations"):
                for r in recs: st.warning(r)
        else:
            st.success("Aucun risque majeur détecté")
        pdf = generate_pdf(student, risk, cluster, recs)
        st.download_button("PDF Rapport", pdf, f"rapport_{id_std}.pdf", "application/pdf")
        st.balloons()

elif page == "Recommandations":
    st.header("Recommandations détaillées")
    rules = load_rules()
    st.dataframe(rules)

elif page == "Historique":
    st.header("Historique")
    if "hist" in st.session_state and not st.session_state.hist.empty:
        st.dataframe(st.session_state.hist.style.format({"Risque": "{:.1%}"}))
        fig = px.bar(st.session_state.hist, x="ID_Étudiant", y="Risque", color="Risque",
                     color_continuous_scale=["#4CAF50", "#F44336"], title="Risques par étudiant")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune donnée")

elif page == "Rapport":
    st.header("Rapport global")
    if "hist" in st.session_state and not st.session_state.hist.empty:
        csv = st.session_state.hist.to_csv(index=False).encode()
        st.download_button("CSV Complet", csv, "rapport_abandon.csv", "text/csv")
    with st.expander("Méthodologie"):
        st.write("K-Means + Random Forest + Apriori")