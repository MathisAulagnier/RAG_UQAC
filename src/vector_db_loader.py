from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import (
    OllamaEmbeddings,
)
from langchain_community.vectorstores import Chroma
from enum import Enum
import os
import shutil
import argparse

CHROMA_PATH = "chroma"
DATA_PATH = "scrapping/output_UQAC_Website"



def load_documents():
    """
    Charge les documents depuis le répertoire spécifié.
    """
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Chargé {len(documents)} documents.")
    return documents

def split_text(documents: list[Document]):
    """
    Découpe les documents en chunks plus petits.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Découpé {len(documents)} documents en {len(chunks)} chunks.")

    if chunks:  # Affiche un exemple de chunk pour vérification
        document = chunks[0]
        print("\nExemple de chunk:")
        print("Contenu:", document.page_content[:200], "...")
        print("Métadonnées:", document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document], ):
    """
    Sauvegarde les chunks dans une base de données Chroma.
    """
    if not chunks:
        print("Aucun chunk à sauvegarder.")
        return

    # Nettoyer la base existante
    if os.path.exists(CHROMA_PATH):
        print(f"Suppression de l'ancienne base de données: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    print(f"Création de la base de données avec Ollama")
    embeddings = get_embeddings()
    
    try:
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Sauvegardé {len(chunks)} chunks dans {CHROMA_PATH}.")
        
        # Test simple de la base de données
        if chunks:
            test_query = chunks[0].page_content[:50]  # Utilise le début du premier chunk
            results = db.similarity_search(test_query, k=1)
            print("\nTest de recherche réussi ✓")
            
    except Exception as e:
        print(f"Erreur lors de la création de la base de données: {str(e)}")
        raise

def main():
    print(f"Démarrage avec Ollama embeddings")
    generate_data_store()

if __name__ == "__main__":
    main()
