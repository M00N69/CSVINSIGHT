import streamlit as st
import pandas as pd
import plotly.express as px
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai import Agent
from pandasai.responses.response_parser import ResponseParser

# Dictionnaire pour stocker les dataframes extraits
data = {}

def main():
    st.set_page_config(page_title="Analyse de données avec IA", page_icon="🤖")
    st.title("Explorez vos données et posez vos questions avec Google GenAI 🤖")

    # Téléchargement de fichier CSV ou Excel
    st.subheader("Téléchargez un fichier CSV ou Excel pour commencer l'analyse")
    file_upload = st.file_uploader("Téléchargez votre fichier", type=['csv', 'xls', 'xlsx'])

    if file_upload:
        try:
            # Extraire les données du fichier
            data = extraire_dataframes(file_upload)
            df_name = st.selectbox("Sélectionnez un tableau de données :", options=list(data.keys()))
            df = data[df_name]
            st.dataframe(df)

            # Affichage d'un graphique avec Plotly
            colonne = st.selectbox("Sélectionnez une colonne pour visualiser les données", df.columns)
            fig = px.histogram(df, x=colonne, title=f"Distribution de {colonne}")
            st.plotly_chart(fig)

            # Intégration de l'IA pour analyser les données
            question = st.text_input("Posez une question à propos de vos données")
            if question:
                llm = obtenir_llm()  # Fonction pour obtenir l'accès à Google GenAI
                if llm:
                    agent = Agent(llm=llm)
                    sdf = SmartDataframe(df)
                    reponse = analyser_donnees(agent, sdf, question)
                    if reponse:
                        st.write(reponse)

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

# Fonction pour extraire les dataframes du fichier téléchargé
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

# Fonction pour configurer Google GenAI avec les secrets de Streamlit
def obtenir_llm():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        return GoogleGemini(api_key=api_key)
    except KeyError:
        st.error("Clé API manquante dans les secrets de Streamlit.")
        return None

# Fonction pour analyser les données avec SmartDataframe et Agent
def analyser_donnees(agent, sdf, question):
    try:
        response = agent.chat(question, sdf)
        return response
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des données : {e}")
        return None

if __name__ == "__main__":
    main()
