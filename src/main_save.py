from save_chunk import save_chunks_to_chroma
from create_chunk import create_chunks_from_file

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma



def main():
    # Initialisation du modèle d'embedding
    embeddings = OllamaEmbeddings(model="llama3", )
    
    # Chemin vers votre fichier markdown
    file_path = "scrapping/output_UQAC_Website/annexe-4-procedure-pour-le-partage-de-linformation-aux-fonds-de-recherche-du-quebec.md"

    # Création des chunks à partir du fichier
    chunks = create_chunks_from_file(file_path)
    
    # Traitement des chunks et stockage dans la base de données Chroma
    nb_chunks = save_chunks_to_chroma(chunks, embeddings)
    print(f"Nombre de chunks stockés : {nb_chunks}")

    
if __name__ == "__main__":
    main()
