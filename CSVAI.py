import streamlit as st
import pandas as pd
import plotly.express as px
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai.connectors import PandasConnector

# Clé API (Assurez-vous de la stocker dans les secrets de Streamlit Cloud)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

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
            prompt = st.text_input("Posez une question à propos de vos données")
            
            if st.button("Analyser") and prompt.strip():
                try:
                    # Préparer le contexte avec des informations sur les données
                    df_info = df.describe(include='all').to_string()
                    prompt_with_context = f"Voici un résumé des colonnes du DataFrame :\n{df_info}\n\n{prompt}"
                    
                    # Utilisation de Google Gemini via PandasAI pour analyser les données
                    llm = GoogleGemini(api_key=GOOGLE_API_KEY)
                    connector = PandasConnector({"original_df": df})
                    sdf = SmartDataframe(connector, {"enable_cache": False}, config={"llm": llm})
                    
                    # Analyser les données avec la question donnée
                    response = sdf.chat(prompt_with_context)
                    st.write("Réponse :")
                    st.write(response)

                    # Afficher le code exécuté
                    st.markdown("### Code exécuté par PandasAI :")
                    st.code(sdf.last_code_executed)

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")

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
        for feuille en xls.sheet_names:
            dfs[feuille] = pd.read_excel(fichier, sheet_name=feuille)
    return dfs

if __name__ == "__main__":
    main()
