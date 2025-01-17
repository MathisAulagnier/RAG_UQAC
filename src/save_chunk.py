import re

from create_chunk import create_chunks_from_file
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma

def extract_url_from_header(text):
    """Extrait l'URL du premier chunk (header)"""
    match = re.match(r'#\s*(https?://[^\s]+)', text)
    if match:
        return match.group(1)
    return None


def save_chunks_to_chroma(chunks, embeddings, persist_directory="./chroma_db"):
    """
    Définie les chunks d'un fichier markdown [create_chunk.py]
    Stocke les chunks dans une base de données Chroma [save_chunk.py]
    """


    # Extraction de l'URL du premier chunk
    url = extract_url_from_header(chunks[0])
    if url is None:
        raise ValueError("Impossible de trouver l'URL dans le header du fichier")
        
    # On ne garde pas le premier chunk qui contient uniquement l'URL
    chunks = chunks[1:]

    # Préparation des métadonnées pour chaque chunk
    metadatas = [{"source": url} for _ in chunks]

    # Création ou chargement de la base de données Chroma
    vectorstore = Chroma(
        collection_name="UQAC_documents",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Ajout des documents avec leurs métadonnées
    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas
    )

    # Persistance des données
    vectorstore.persist()

    return len(chunks)

