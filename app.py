# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle
model = joblib.load('model.pkl')

# Titre
st.title("🧮 Prédiction de détention de compte bancaire")

# Champs du formulaire
country = st.selectbox("Pays", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])  # adapter si besoin
year = st.selectbox("Année", [2016, 2017, 2018])
location_type = st.selectbox("Type de lieu", ['Rural', 'Urbain'])
cellphone_access = st.selectbox("Accès à un téléphone portable", ['Oui', 'Non'])
household_size = st.slider("Taille du foyer", 1, 20, 4)
age = st.slider("Âge du répondant", 10, 100, 30)
gender = st.selectbox("Genre", ['Homme', 'Femme'])
relationship = st.selectbox("Relation au chef de famille", ['Head of Household', 'Spouse', 'Child', 'Other relative'])
marital_status = st.selectbox("Statut marital", ['Single/Never Married', 'Married/Living together', 'Widowed'])
education_level = st.selectbox("Niveau d’éducation", [
    'No formal education', 'Primary education', 'Secondary education',
    'Tertiary education', 'Vocational/Specialised training'
])
job_type = st.selectbox("Type d'emploi", [
    'Self employed', 'Government Dependent', 'Formally employed Private',
    'Informally employed', 'Formally employed Government', 'Farming and Fishing', 'Other Income'
])

# Encodeurs fictifs (les vrais doivent correspondre à ton LabelEncoder)
# Remplace ces dictionnaires par les bons encodages après avoir imprimé ceux utilisés
encoders = {
    'country': {'Kenya': 0, 'Rwanda': 1, 'Tanzania': 2, 'Uganda': 3},
    'location_type': {'Rural': 0, 'Urbain': 1},
    'cellphone_access': {'Non': 0, 'Oui': 1},
    'gender': {'Homme': 1, 'Femme': 0},
    'relationship': {'Head of Household': 0, 'Spouse': 1, 'Child': 2, 'Other relative': 3},
    'marital_status': {
        'Single/Never Married': 0,
        'Married/Living together': 1,
        'Widowed': 2
    },
    'education_level': {
        'No formal education': 0,
        'Primary education': 1,
        'Secondary education': 2,
        'Vocational/Specialised training': 3,
        'Tertiary education': 4
    },
    'job_type': {
        'Self employed': 0,
        'Government Dependent': 1,
        'Formally employed Private': 2,
        'Informally employed': 3,
        'Formally employed Government': 4,
        'Farming and Fishing': 5,
        'Other Income': 6
    }
}

# Préparer les données dans le bon ordre
input_data = pd.DataFrame([{
    'country': encoders['country'][country],
    'year': year,
    'location_type': encoders['location_type'][location_type],
    'cellphone_access': encoders['cellphone_access'][cellphone_access],
    'household_size': household_size,
    'age_of_respondent': age,
    'gender_of_respondent': encoders['gender'][gender],
    'relationship_with_head': encoders['relationship'][relationship],
    'marital_status': encoders['marital_status'][marital_status],
    'education_level': encoders['education_level'][education_level],
    'job_type': encoders['job_type'][job_type]
}])

# Bouton de prédiction
if st.button("Prédire"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Cette personne est susceptible d'avoir un compte bancaire.")
    else:
        st.warning("❌ Cette personne n'est probablement pas détentrice d'un compte bancaire.")

