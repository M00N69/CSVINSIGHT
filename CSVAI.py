import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO
import datetime
from Levenshtein import distance
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai.responses.response_parser import ResponseParser

# Dictionnaire pour stocker les dataframes extraits
data = {}

def main():
    st.set_page_config(page_title="Exploration avec IA", page_icon="🤖")
    st.title("Discutez avec vos données grâce à Google GenAI 🤖")

    # Téléchargement de fichier CSV ou Excel
    st.subheader("Téléchargez un fichier CSV ou Excel")
    file_upload = st.file_uploader("Téléchargez votre fichier", type=['csv', 'xls', 'xlsx'])
    
    if file_upload:
        try:
            # Extraire les données
            data = extraire_dataframes(file_upload)
            df_name = st.selectbox("Sélectionnez un tableau de données à partir de votre fichier :", options=list(data.keys()))
            df = data[df_name]
            st.dataframe(df)

            # Affichage d'un graphique
            colonne = st.selectbox("Sélectionnez une colonne pour visualiser les données", df.columns)
            fig = px.histogram(df, x=colonne, title=f"Distribution de {colonne}")
            st.plotly_chart(fig)

            # Intégration de l'IA pour analyser les données
            question = st.text_input("Posez une question à propos de vos données")
            if question:
                llm = obtenir_llm()  # Récupérer l'API Google
                if llm:
                    sdf = SmartDataframe(df, llm=llm)
                    reponse = analyser_donnees(sdf, question)
                    if reponse:
                        st.write(reponse)

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

# Fonction pour extraire les dataframes à partir du fichier téléchargé
def extraire_dataframes(fichier):
    dfs = {}
    extension = fichier.name.split('.')[-1]

    if extension == 'csv':
        nom_df = fichier.name.split('.')[0]
        dfs[nom_df] = pd.read_csv(fichier)
    elif extension in ['xls', 'xlsx']:
        xls = pd.ExcelFile(fichier)
        for feuille in xls.sheet_names:
            dfs[feuille] = pd.read_excel(fichier, sheet_name=feuille)
    return dfs

# Fonction pour configurer Google GenAI avec les secrets Streamlit
def obtenir_llm():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        return GoogleGemini(api_key=api_key)
    except KeyError:
        st.error("Clé API manquante dans les secrets.")
        return None

# Fonction pour analyser les données avec SmartDataframe
def analyser_donnees(sdf, question):
    try:
        response = sdf.chat(question, response_parser=ResponseParser())
        return response
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
        return None

if __name__ == "__main__":
    main()
