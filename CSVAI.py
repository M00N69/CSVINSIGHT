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

            # Intégration de l'IA pour analyser les données
            prompt = st.text_input("Posez une question à propos de vos données")

            if st.button("Analyser") and prompt.strip():
                try:
                    # Fournir un contexte explicite à PandasAI
                    contexte = generer_contexte(df)
                    prompt_with_context = f"{contexte}\n\n{prompt}\nMerci de fournir une analyse détaillée, des explications textuelles et éviter de générer des graphiques enregistrés dans des fichiers."

                    # Utilisation de Google Gemini via PandasAI pour analyser les données
                    llm = GoogleGemini(api_key=GOOGLE_API_KEY)
                    connector = PandasConnector({"original_df": df})
                    sdf = SmartDataframe(connector, {"enable_cache": False}, config={"llm": llm})

                    # Analyser les données avec le contexte et la question donnée
                    response = sdf.chat(prompt_with_context)

                    st.write("Réponse de l'IA :")
                    st.write(response)

                    # Gestion manuelle des graphiques si la question concerne les causes de fraudes et les catégories
                    if "fraudes" in prompt and "catégories" in prompt:
                        st.write("Voici le graphique des causes de fraudes et des catégories de produits touchées:")

                        # Extraire les données pour les adulterants et les catégories
                        top_adulterants = df['adulterant'].value_counts().nlargest(5)
                        top_categories = df['category'].value_counts().nlargest(5)

                        # Créer les graphiques avec matplotlib
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                        # Graphique des adulterants (causes de fraudes)
                        ax1.bar(top_adulterants.index, top_adulterants.values)
                        ax1.set_xlabel('Adulterant')
                        ax1.set_ylabel('Count')
                        ax1.set_title('Top 5 Adulterants')
                        ax1.tick_params(axis='x', rotation=45)

                        # Graphique des catégories (produits touchés)
                        ax2.bar(top_categories.index, top_categories.values)
                        ax2.set_xlabel('Product Category')
                        ax2.set_ylabel('Count')
                        ax2.set_title('Top 5 Product Categories')
                        ax2.tick_params(axis='x', rotation=45)

                        # Ajustement de la mise en page
                        plt.tight_layout()

                        # Afficher les graphiques dans Streamlit
                        st.pyplot(fig)

                    # Afficher le code exécuté par PandasAI
                    st.markdown("### Code exécuté par PandasAI :")
                    st.code(sdf.last_code_executed)

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

# Fonction pour générer un contexte complet et détaillé pour PandasAI
def generer_contexte(df):
    """Génère une description complète du DataFrame, incluant types de colonnes, valeurs uniques, et statistiques."""
    description = df.describe(include='all').to_string()
    colonnes_info = []
    for col in df.columns:
        info = f"Colonne '{col}' - Type : {df[col].dtype}, Valeurs uniques : {df[col].nunique()}"
        colonnes_info.append(info)
    colonnes_info_str = "\n".join(colonnes_info)
    contexte = f"Voici une description des données disponibles :\n{colonnes_info_str}\n\nStatistiques :\n{description}"
    return contexte

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
