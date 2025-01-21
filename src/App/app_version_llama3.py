import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

# Initialisation de l'historique de conversation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Configuration de la page Streamlit
st.set_page_config(page_title="RAG Chatbot UQAC", page_icon="🤖")
st.title("Chatbot UQAC basé sur RAG")
st.write("Posez une question en rapport avec le guide de gestion de l'UQAC.")

# Fonction pour charger la base de données Chroma existante
@st.cache_resource
def load_chroma_db(persist_directory="./../chroma_db"):
    """Charge la base Chroma existante sans la recréer."""
    if not os.path.exists(persist_directory):
        st.error(f"Le répertoire spécifié pour la base Chroma n'existe pas : {persist_directory}")
        return None
    retriever = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="llama3")
    ).as_retriever(search_kwargs={"k": 3})  # Retourne les 3 documents les plus proches
    return retriever

# Charger le retriever
retriever = load_chroma_db()

if retriever is None:
    st.stop()

# Chargement du modèle Llama 3 depuis Hugging Face
llama_model_name = "meta-llama/Llama-3.1-8B"  # Remplacez ceci avec le nom exact du modèle Llama 3 sur Hugging Face
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
model = AutoModelForCausalLM.from_pretrained(llama_model_name, device_map="auto")

# Créer un pipeline HuggingFace pour utiliser Llama 3
def hf_generate(inputs):
    inputs = tokenizer(inputs['text'], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(inputs["input_ids"], max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Utiliser HuggingFacePipeline
chat_model = HuggingFacePipeline(pipeline=hf_generate)

# Configuration de la chaîne RAG
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Interface utilisateur
user_input = st.text_input("Votre question :", key="user_input")

if user_input:
    with st.spinner("Recherche en cours..."):
        # Récupérer la réponse
        response = qa_chain({"question": user_input})

        # Construire une réponse claire avec les sources
        st.success("Réponse générée :")
        st.write(response["answer"])

        # Vérifier et afficher les sources
        if response.get("source_documents"):
            st.write("### Sources utilisées :")
            for source_doc in response["source_documents"]:
                source_url = source_doc.metadata.get('source', 'URL inconnue')
                st.markdown(f"- **Source :** [{source_url}]({source_url})")

        # Afficher l'historique de conversation
        st.write("### Historique des échanges")
        for message in memory.chat_memory.messages:
            if message["type"] == "user":
                st.markdown(f"**Vous** : {message['content']}")
            else:
                st.markdown(f"**Chatbot** : {message['content']}")
