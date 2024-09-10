import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai.connectors import PandasConnector
import io

# Clé API (Assurez-vous de la stocker dans les secrets de Streamlit Cloud)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# Dictionnaire pour stocker les dataframes extraits
data = {}

def main():
    st.set_page_config(page_title="Analyse de données avec IA", page_icon="🐼")
    st.title("Explorez vos données et posez vos questions avec Google GenAI 🐼")

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

            # Vérifier l'intégrité des données
            verifier_integrite_donnees(df)

            # Affichage d'un graphique avec matplotlib (sans passer par le fichier)
            prompt = st.text_input("Posez une question à propos de vos données")
            
            if st.button("Analyser") and prompt.strip():
                try:
                    # Utilisation de Google Gemini via PandasAI pour analyser les données
                    llm = GoogleGemini(api_key=GOOGLE_API_KEY)
                    connector = PandasConnector({"original_df": df})
                    sdf = SmartDataframe(connector, {"enable_cache": False}, config={"llm": llm})
                    
                    # Analyser les données avec la question donnée
                    response = sdf.chat(prompt)

                    if isinstance(response, dict) and response.get('type') == 'plot':
                        # Simuler une analyse manuelle avec matplotlib et afficher directement le graphique
                        # Créer le graphique correspondant à la réponse
                        top_adulterants = df['adulterant'].value_counts().nlargest(5)
                        top_categories = df['category'].value_counts().nlargest(5)
                        
                        # Créer les graphiques
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                        # Graphique des fraudes (adulterants)
                        ax1.bar(top_adulterants.index, top_adulterants.values)
                        ax1.set_xlabel('Adulterant')
                        ax1.set_ylabel('Count')
                        ax1.set_title('Top 5 Adulterants')
                        ax1.tick_params(axis='x', rotation=45)

                        # Graphique des catégories de produits (categories)
                        ax2.bar(top_categories.index, top_categories.values)
                        ax2.set_xlabel('Product Category')
                        ax2.set_ylabel('Count')
                        ax2.set_title('Top 5 Product Categories')
                        ax2.tick_params(axis='x', rotation=45)

                        # Ajuster la mise en page
                        plt.tight_layout()

                        # Afficher le graphique dans Streamlit
                        st.pyplot(fig)

                    else:
                        st.warning("Aucun graphique n'a été généré.")
                    
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
        for feuille in xls.sheet_names:
            dfs[feuille] = pd.read_excel(fichier, sheet_name=feuille)
    return dfs

# Fonction pour vérifier l'intégrité des données avant l'analyse
def verifier_integrite_donnees(df):
    """Vérifie les types de colonnes et prépare les données avant de les analyser."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Si la colonne est numérique, vérifier s'il y a des valeurs manquantes
            if df[col].isnull().sum() > 0:
                st.warning(f"La colonne '{col}' contient des valeurs manquantes. Elles seront ignorées dans l'analyse.")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            st.info(f"La colonne '{col}' est de type date.")
        elif pd.api.types.is_bool_dtype(df[col]):
            st.info(f"La colonne '{col}' est booléenne (True/False).")
        elif pd.api.types.is_object_dtype(df[col]):
            st.info(f"La colonne '{col}' est de type 'object'. Elle sera convertie en 'category' pour optimiser l'analyse.")
            df[col] = df[col].astype('category')
        else:
            st.warning(f"Le type de données de la colonne '{col}' est inconnu ou non pris en charge.")

if __name__ == "__main__":
    main()
