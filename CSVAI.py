import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

            # Vérifier l'intégrité des données sans convertir les types
            verifier_integrite_donnees(df)

            # Générer un contexte pour orienter l'utilisateur
            contexte, suggestions = generer_contexte_et_suggestions(df)

            st.markdown("### Contexte des données :")
            st.write(contexte)

            st.markdown("### Suggestions de questions :")
            st.write(suggestions)

            # Input utilisateur : la question à poser à l'IA
            prompt = st.text_input("Posez une question à propos de vos données ou sélectionnez une question suggérée", value=suggestions[0] if suggestions else '')

            if st.button("Analyser") and prompt.strip():
                try:
                    # Fournir un contexte explicite à PandasAI
                    prompt_with_context = f"{contexte}\n\n{prompt}\nMerci de fournir une analyse textuelle détaillée."

                    # Utilisation de Google Gemini via PandasAI pour analyser les données
                    llm = GoogleGemini(api_key=GOOGLE_API_KEY)
                    connector = PandasConnector({"original_df": df})
                    sdf = SmartDataframe(connector, {"enable_cache": False}, config={"llm": llm})

                    # Analyser les données avec le contexte et la question donnée
                    response = sdf.chat(prompt_with_context)

                    st.write("Réponse de l'IA :")
                    st.write(response)

                    # Afficher le code exécuté par PandasAI
                    st.markdown("### Code exécuté par PandasAI :")
                    st.code(sdf.last_code_executed)

                    # Gestion manuelle des graphiques
                    st.write("Voici une illustration graphique des données :")

                    for col in df.select_dtypes(include=['number']).columns:
                        fig, ax = plt.subplots()
                        df[col].plot(kind='hist', ax=ax, title=f"Distribution de {col}")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

# Fonction pour générer un contexte complet et proposer des suggestions de questions basées sur les colonnes
def generer_contexte_et_suggestions(df):
    """Génère un résumé des colonnes et propose des suggestions de questions en fonction des données disponibles."""
    description = df.describe(include='all').to_string()
    colonnes_info = []
    suggestions = []

    for col in df.columns:
        info = f"Colonne '{col}' - Type : {df[col].dtype}, Valeurs uniques : {df[col].nunique()}"
        colonnes_info.append(info)

        # Proposer des questions en fonction du type de colonne
        if pd.api.types.is_numeric_dtype(df[col]):
            suggestions.append(f"Quelles sont les statistiques de base de la colonne '{col}' ?")
            suggestions.append(f"Montre-moi un graphique de la distribution de '{col}' ?")
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            suggestions.append(f"Quelles sont les catégories principales de la colonne '{col}' ?")
            suggestions.append(f"Comment la colonne '{col}' influence-t-elle les autres variables ?")

    colonnes_info_str = "\n".join(colonnes_info)
    contexte = f"Voici un résumé des colonnes de vos données :\n{colonnes_info_str}\n\nStatistiques :\n{description}"

    return contexte, suggestions

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

# Fonction pour vérifier l'intégrité des données avant l'analyse sans conversion automatique
def verifier_integrite_donnees(df):
    """Vérifie les types de colonnes et prépare les données avant de les analyser sans changer les types."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().sum() > 0:
                st.warning(f"La colonne '{col}' contient des valeurs manquantes. Elles seront ignorées dans l'analyse.")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            st.info(f"La colonne '{col}' est de type date.")
        elif pd.api.types.is_bool_dtype(df[col]):
            st.info(f"La colonne '{col}' est booléenne (True/False).")
        elif pd.api.types.is_object_dtype(df[col]):
            st.info(f"La colonne '{col}' est de type 'object'. Elle ne sera pas convertie.")
        else:
            st.warning(f"Le type de données de la colonne '{col}' est inconnu ou non pris en charge.")

if __name__ == "__main__":
    main()
