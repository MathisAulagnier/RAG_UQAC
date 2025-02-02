import os
import sys
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# Ajouter le dossier parent au chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))

from create_chunk import create_chunks_from_file
from save_chunk import save_chunks_to_chroma


# Initialisation de l'historique de conversation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "summary_history" not in st.session_state:
    st.session_state["summary_history"] = []
summary_history = st.session_state["summary_history"]

empty_history = []

# Configuration de la page Streamlit
st.set_page_config(page_title="ChatBot UQAC (Architure RAG)", page_icon="🤖")
st.title("Chatbot spécialisé de la gestion à l'UQAC")
st.write("Posez une question en rapport avec le guide de gestion de l'UQAC.")

# Bouton de réinitialisation de la conversation
if st.sidebar.button("🆕 Nouveau Chat (attention vous allez perdre le chat actuel)"):
    st.session_state["chat_history"] = []
    st.session_state["summary_history"] = []
    st.session_state["user_input"] = ""
    st.rerun()


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
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Configuration de la chaîne RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    output_key="answer"
)

# Interface utilisateur
st.write("### Historique de la conversation :")
for message in st.session_state["chat_history"]:
    if isinstance(message, HumanMessage):
        st.write(f"**Vous**: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**Chatbot**: {message.content}")

user_input = st.text_input("Votre question :", key="user_input", placeholder="Posez votre question ici...")

if user_input:
    with st.spinner("Recherche en cours..."):
        try:
            # Prompt pour la première réponse
            prompt = (
                    "Instructions: Summarize the enclosed information coherently to give a direct, well-supported answer, citing your sources without any visible preliminary thought process to answer the question."
                    "You must absolutely answer the question! "
                    "Your response must begin with:"
                    f'"Question: {user_input}\n <br> \n'
                    f'"To answer the question {user_input}, several key factors need to be considered: 1. ...\n\n"'
                    "Recent conversation:\n"
                    + (f"1. {summary_history[-2]} \n" if len(summary_history) > 1 else "None")
                    + (f"2. {summary_history[-1]} \n" if len(summary_history) > 0 else "")
            )

            # Afficher le prompt
            print("Prompt envoyé au modèle : ",prompt)

            # Envoyer la requête au modèle
            response = qa_chain.invoke({"question": prompt, "chat_history": empty_history})
            chatbot_response = response["answer"]

            # Deuxième requête au LLM pour s'assurer que la réponse est bien en français
            translation_prompt = (
                "Translate this text into French if it's in English without further comment like 'Voici le texte traduit en français :' Give me just the translation of the text. If it's already in French, return it as is without modifying anything: "
                f'"{chatbot_response}"'
            )
            translation_response = llm.invoke(translation_prompt)

            if len(summary_history) >= 2:
                # Troisième requête au LLM pour faire un résumé de toutes les questions de l'utilisateur
                question_prompt = (
                    "You must summarize these two questions into a single one: "
                    f'First question: "{user_input}"\n'
                    f'Second question: "{summary_history[-2]}"'
                )
                print("question_prompt : ", question_prompt)
                question_summarized = llm.invoke(question_prompt)

                # Quatrième requête au LLM pour faire un résumé de toutes les réponses du chatbot
                answer_prompt = (
                    "You must summarize these two answers into a single one: "
                    f'First answer: "{translation_response}"\n'
                    f'Second answer: "{summary_history[-1]}"'
                )
                print("answer_prompt : ", answer_prompt)
                answer_summarized = llm.invoke(answer_prompt)

                # Cinquième requête au LLM pour s'assurer que le résumé des questions de l'utilisateur est bien en français
                translation_prompt = (
                    "Translate this text into French if it's in English without further comment like 'Voici le texte traduit en français :' Give me just the translation of the text. If it's already in French, return it as is without modifying anything: "
                    f'"{question_summarized}"'
                )
                translation_question_summarized = llm.invoke(translation_prompt)

                # Sixième requête au LLM pour s'assurer que le résumé des réponses du chatbot est bien en français
                translation_prompt = (
                    "Translate this text into French if it's in English without further comment like 'Voici le texte traduit en français :' Give me just the translation of the text. If it's already in French, return it as is without modifying anything: "
                    f'"{answer_summarized}"'
                )
                translation_answer_summarized = llm.invoke(translation_prompt)

                summary_history.append(translation_question_summarized)
                summary_history.append(translation_answer_summarized)
                print("\nsummarized :\n")
                print(len(summary_history))
                print(translation_question_summarized)
                print(translation_answer_summarized)
            else:
                print(len(summary_history))
                summary_history.append(user_input)
                summary_history.append(translation_response)
                print("\nnon_summarized :\n")
                print(len(summary_history))
                print(user_input)
                print(translation_response)

            # Mettre à jour l'historique de session
            st.session_state["chat_history"].append(HumanMessage(content=user_input))
            st.session_state["chat_history"].append(AIMessage(content=translation_response))

            # Afficher la réponse
            st.success("Réponse générée :")
            st.write(translation_response)

            add_in_db = False
            # Afficher les documents sources
            if response.get("source_documents"):
                st.write("### Sources utilisées :")
                add_in_db = True
                for idx, doc in enumerate(response["source_documents"], 1):
                    st.write(f"#### Document {idx}")
                    st.markdown(f"📜 **Contenu du chunk :**\n\n{doc.page_content}")
                    source = doc.metadata.get('source', 'Source non spécifiée')
                    if source == "https://www.chatbot.ca/chatbot_file" :
                        add_in_db = False
                        st.markdown(f"- **Source :** Réponse déjà générée par le chatbot dans une autre conversation")
                        print("Source chatbot détecté\n\n")
                    else :
                        st.markdown(f"- **Source :** {source}")
            else:
                st.warning("Aucun document source n'a été trouvé pour cette requête.")

            if add_in_db:
                print("ad in cause : ", add_in_db)
                # Créer ou écraser le fichier .md
                file_path = "conversation.md"
                if os.path.exists(file_path):
                    os.remove(file_path)

                # Créer un fichier Markdown avec la question et la réponse
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("# https://www.chatbot.ca/chatbot_file\n\n")
                    f.write(f"## La question de l'utilisateur :\n{user_input}\n\n")
                    f.write(f"## Réponse du chatbot :\n{translation_response}\n")

                # Création des chunks pour le fichier
                chunks = create_chunks_from_file("conversation.md")

                # Sauvegarde des chunks dans Chroma
                embeddings = OllamaEmbeddings(
                    model="llama3",
                    base_url="http://localhost:11434"
                )

                if not chunks:
                    st.error("Aucun chunk généré.")
                else:
                    # Vérifie que chaque chunk a un embedding associé
                    for chunk in chunks:
                        embedding = embeddings.embed_documents([chunk])
                        if not embedding:
                            st.error(f"Aucun embedding généré pour le chunk : {chunk}")

                # Sauvegarde des chunks dans Chroma
                nb_chunks = save_chunks_to_chroma(
                    chunks=chunks,
                    embeddings=embeddings,
                    persist_directory="./../chroma_db"
                )

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
