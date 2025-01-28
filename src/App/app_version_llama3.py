import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import chromadb
from chromadb.config import Settings

# Initialisation de l'historique de conversation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Configuration de la page Streamlit
st.set_page_config(page_title="RAG Chatbot UQAC", page_icon="🤖")
st.title("Chatbot UQAC basé sur RAG")
st.write("Posez une question en rapport avec le guide de gestion de l'UQAC.")


# Fonction pour charger la base de données Chroma existante
@st.cache_resource
def load_chroma_db(
        persist_directory="../chroma_db"):
    try:
        if not os.path.exists(persist_directory):
            st.error(f"Le répertoire spécifié pour la base Chroma n'existe pas : {persist_directory}")
            return None

        embeddings = OllamaEmbeddings(
            model="llama3",
            base_url="http://localhost:11434"
        )

        # Chargement avec le nom correct de la collection
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="UQAC_documents"  # Nom exact de votre collection
        )

        # Afficher les informations de débogage
        collection_count = db._collection.count()
        st.write(f"Nombre de documents dans la collection : {collection_count}")

        if collection_count == 0:
            st.error("La collection est vide !")
            return None

        # Créer le retriever avec des paramètres simples
        retriever = db.as_retriever(search_kwargs={"k": 3})

        return retriever

    except Exception as e:
        st.error(f"Erreur lors du chargement de la base Chroma : {str(e)}")
        return None


# Charger le retriever
retriever = load_chroma_db()

if retriever is None:
    st.stop()

# Configurer le LLM avec Ollama
llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Configuration de la mémoire
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Configuration de la chaîne RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# Ajout d'une instruction initiale pour répondre en français
memory.chat_memory.add_ai_message("Je répondrai en français à toutes les questions en utilisant les documents fournis.")

# Interface utilisateur
user_input = st.text_input("Votre question :", key="user_input")

if user_input:
    with st.spinner("Recherche en cours..."):
        try:
            # Préparer le prompt
            prompt = (
                "Sur la base des documents extraits du guide de gestion de l'UQAC, "
                "faites une synthèse cohérente pour répondre en français à cette question : "
                f"{user_input}\n"
                "Basez votre réponse uniquement sur les informations présentes dans les documents fournis."
            )

            # Récupérer la réponse
            response = qa_chain.invoke({"question": prompt})

            # Afficher la réponse
            st.success("Réponse générée :")
            st.write(response["answer"])

            # Afficher les documents sources
            if response.get("source_documents"):
                st.write("### Sources utilisées :")
                for idx, doc in enumerate(response["source_documents"], 1):
                    st.write(f"#### Document {idx}")
                    st.write(f"**Contenu :** {doc.page_content}")
                    st.write(f"**Métadonnées :** {doc.metadata}")
                    source = doc.metadata.get('source', 'Source non spécifiée')
                    st.markdown(f"- **Source :** {source}")
            else:
                st.warning("Aucun document source n'a été trouvé pour cette requête.")

            # Afficher l'historique
            st.write("### Historique des échanges")
            for message in memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    st.markdown(f"**Vous :** {message.content}")
                elif isinstance(message, AIMessage):
                    st.markdown(f"**Assistant :** {message.content}")

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
