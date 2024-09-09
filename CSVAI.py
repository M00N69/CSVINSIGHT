import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.llm import GoogleGemini
from pandasai.responses.response_parser import ResponseParser
from io import BytesIO

# Dictionnaire pour stocker les dataframes extraits
data = {}

def main():
    st.set_page_config(page_title="Discussion avec vos données", page_icon="📊")
    st.title("Discutez avec vos données grâce à Google GenAI 📊")

    # Configuration de la barre latérale
    with st.sidebar:
        st.title("Configuration ⚙️")
        
        # Téléchargement de fichier
        st.subheader("Téléchargement des données 📝")
        file_upload = st.file_uploader("Téléchargez votre fichier CSV ou Excel", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
        st.markdown(":green[*Veuillez vous assurer que la première ligne contient les noms des colonnes.*]")

    if file_upload:
        # Extraction des dataframes
        try:
            data = extraire_dataframes(file_upload)
            df_name = st.selectbox("Sélectionnez un tableau de données à partir de votre fichier :", tuple(data.keys()), index=0)
            st.dataframe(data[df_name])
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return
        
        # Configuration de Google Gemini avec st.secrets
        llm = obtenir_llm()
        if llm:
            # Utilisation de SmartDataframe pour interagir avec les données
            sdf = SmartDataframe(data[df_name], connector=PandasConnector(), llm=llm)
            
            # Discussion avec les données
            fenetre_chat(sdf, data[df_name])
        else:
            st.error("Clé API manquante ou invalide dans les secrets de Streamlit.")
    else:
        st.warning("Veuillez télécharger un fichier CSV ou Excel pour commencer.")

# Fonction pour configurer Google GenAI avec les secrets Streamlit
def obtenir_llm():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        return GoogleGemini(api_key=api_key)
    except KeyError:
        st.error("Aucune clé API trouvée dans les secrets de Streamlit.")
        return None

# Fonction pour gérer l'interface de chat et fournir une analyse graphique
def fenetre_chat(sdf, df):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique du chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message["response"])
            elif 'graph' in message:
                st.pyplot(message['graph'])
    
    # Champ de saisie pour les questions de l'utilisateur
    question_utilisateur = st.chat_input("Que souhaitez-vous demander à propos des données ? Vous pouvez aussi demander un graphique.")

    if question_utilisateur:
        with st.chat_message("utilisateur"):
            st.markdown(question_utilisateur)
        st.session_state.messages.append({"role": "utilisateur", "question": question_utilisateur})

        try:
            with st.spinner("Analyse des données en cours..."):
                # Utilisation du SmartDataframe pour analyser les questions de l'utilisateur
                reponse = sdf.chat(question_utilisateur, response_parser=ResponseParser())

                # Vérifier si la réponse inclut une demande de graphique
                if "graphique" in question_utilisateur.lower():
                    # Générer un graphique (par exemple, histogramme d'une colonne sélectionnée)
                    colonne = st.selectbox("Sélectionnez une colonne pour le graphique", df.columns)
                    fig, ax = plt.subplots()
                    df[colonne].plot(kind="hist", ax=ax)
                    plt.title(f"Histogramme de {colonne}")
                    st.pyplot(fig)

                    # Ajout du graphique et de l'explication à l'historique du chat
                    st.session_state.messages.append({"role": "assistant", "response": reponse, "graph": fig})

                    # Option de téléchargement du graphique
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png")
                    buffer.seek(0)
                    st.download_button(label="Télécharger le graphique en PNG", data=buffer, file_name="graphique.png", mime="image/png")

                    # Sauvegarder l'explication sous forme de fichier texte
                    explication = f"Explication du graphique de {colonne} : {reponse}"
                    st.download_button(label="Télécharger l'analyse en TXT", data=explication, file_name="analyse.txt", mime="text/plain")

                else:
                    # Si aucun graphique n'est demandé, afficher simplement la réponse
                    st.write(reponse)
                    st.session_state.messages.append({"role": "assistant", "response": reponse})
        
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.session_state.messages.append({"role": "assistant", "error": "Impossible de générer une réponse. Veuillez réessayer."})

    # Bouton pour effacer l'historique du chat
    st.sidebar.button("Effacer l'historique du chat", on_click=effacer_historique_chat)

# Fonction pour effacer l'historique du chat
def effacer_historique_chat():
    st.session_state.messages = []

# Fonction pour extraire les dataframes du fichier téléchargé
def extraire_dataframes(fichier):
    dfs = {}
    type_fichier = fichier.name.split('.')[-1]

    if type_fichier == 'csv':
        nom_df = fichier.name.split('.')[0]
        dfs[nom_df] = pd.read_csv(fichier)
    elif type_fichier in ['xls', 'xlsx']:
        xls = pd.ExcelFile(fichier)
        for feuille in xls.sheet_names:
            dfs[feuille] = pd.read_excel(fichier, sheet_name=feuille)
    return dfs

if __name__ == "__main__":
    main()
