import os
import re
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import chromadb
from chromadb.config import Settings

# Fonction pour nettoyer la réponse des balises <think>
def clean_response(text):
    #cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    #cleaned_text = '\n'.join(line.strip() for line in cleaned_text.split('\n') if line.strip())
    return text

# Initialisation de l'historique de conversation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Configuration de la page Streamlit
st.set_page_config(page_title="ChatBot UQAC (Architure RAG)", page_icon="🤖")
st.title("Chatbot spécialisé de la gestion à l'UQAC")
st.write("Posez une question en rapport avec le guide de gestion de l'UQAC.")

# Fonction pour charger la base de données Chroma existante
@st.cache_resource
def load_chroma_db(persist_directory="../chroma_db"):
    try:
        if not os.path.exists(persist_directory):
            st.error(f"Le répertoire spécifié pour la base Chroma n'existe pas : {persist_directory}")
            return None

        embeddings = OllamaEmbeddings(
            model="llama3",
            base_url="http://localhost:11434"
        )

        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="UQAC_documents"
        )

        collection_count = db._collection.count()
        st.write(f"Nombre de documents dans la collection : {collection_count}")

        if collection_count == 0:
            st.error("La collection est vide !")
            return None

        retriever = db.as_retriever(search_kwargs={"k": 10})
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
    model="mistral",
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

# Interface utilisateur
st.write("### Historique de la conversation :")
for message in st.session_state["chat_history"]:
    if isinstance(message, HumanMessage):
        st.write(f"**Vous**: {message.content}")
    elif isinstance(message, AIMessage):
        cleaned_content = clean_response(message.content)
        st.write(f"**Chatbot**: {cleaned_content}")

user_input = st.text_input("Votre question :", key="user_input", placeholder="Posez votre question ici...")

if user_input:
    with st.spinner("Recherche en cours..."):
        try:
            memory.chat_memory.add_user_message(user_input)

            # Prompt pour la première réponse
            prompt = (
                "Instructions: Summarize the enclosed information coherently to give a direct, well-supported answer, citing your sources without any visible preliminary thought process to answer the question."
                "You must absolutely answer the question! "
                "Your response must begin with:"
                f'"Question: {user_input}\n <br> \n'
                f'"To answer the question '"'{user_input}', several key factors need to be considered: 1. ..."'"'
            )

            # 🔹 Première requête au LLM (réponse initiale)
            response = qa_chain.invoke({"question": prompt})
            chatbot_response = clean_response(response["answer"])

            # 🔹 Deuxième requête au LLM pour s'assurer que la réponse est bien en français
            translation_prompt = (
                "Translate this text into French if it's in English. If it's already in French, return it as is without modifying anything: "
                f'"{chatbot_response}"'
            )
            translation_response = llm.invoke(translation_prompt)

            # Nettoyer la réponse traduite
            final_response = clean_response(translation_response)

            # Ajouter la réponse finale à la mémoire
            memory.chat_memory.add_ai_message(final_response)

            # Mettre à jour l'historique de session
            st.session_state["chat_history"].append(HumanMessage(content=user_input))
            st.session_state["chat_history"].append(AIMessage(content=final_response))

            # Afficher la réponse
            st.success("Réponse générée :")
            st.write(final_response)

            # Afficher les documents sources
            if response.get("source_documents"):
                st.write("### Sources utilisées :")
                for idx, doc in enumerate(response["source_documents"], 1):
                    st.write(f"#### Document {idx}")
                    st.markdown(f"📜 **Contenu du chunk :**\n\n{doc.page_content}")
                    source = doc.metadata.get('source', 'Source non spécifiée')
                    st.markdown(f"- **Source :** {source}")
            else:
                st.warning("Aucun document source n'a été trouvé pour cette requête.")

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
