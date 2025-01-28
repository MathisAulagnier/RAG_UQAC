import os
from langchain_ollama import OllamaEmbeddings
import chromadb
import argparse

from create_chunk import create_chunks_from_file
from save_chunk import save_chunks_to_chroma

def process_directory(directory_path, persist_directory="./chroma_db"):
    """
    Traite tous les fichiers markdown d'un répertoire et les stocke dans Chroma
    """
    # Initialisation du client Chroma et suppression de l'ancienne collection
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection("UQAC_documents")
    except:
        pass

    # Initialisation des embeddings
    embeddings = OllamaEmbeddings(model="llama3")
    
    # Compteurs pour les statistiques
    total_files = 0
    total_chunks = 0
    
    # Parcours des fichiers du répertoire
    for filename in os.listdir(directory_path):
        if filename.endswith('.md'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Création des chunks pour le fichier
                chunks = create_chunks_from_file(file_path)
                
                # Sauvegarde des chunks dans Chroma
                nb_chunks = save_chunks_to_chroma(
                    chunks=chunks,
                    embeddings=embeddings,
                    persist_directory=persist_directory
                )
                
                total_files += 1
                total_chunks += nb_chunks
                
                print(f"Traité avec succès: {filename} ({nb_chunks} chunks)")
                
            except Exception as e:
                print(f"Erreur lors du traitement de {filename}: {str(e)}")
    
    return total_files, total_chunks


def main():
    # Configuration du parser d'arguments
    parser = argparse.ArgumentParser(description="Traitement de fichiers markdown pour RAG")
    parser.add_argument(
        "directory",
        help="Chemin vers le répertoire contenant les fichiers markdown"
    )
    parser.add_argument(
        "--persist_dir",
        default="./chroma_db",
        help="Répertoire de persistance pour la base Chroma"
    )
    
    # Parsing des arguments
    args = parser.parse_args()
    
    # Vérification de l'existence du répertoire
    if not os.path.isdir(args.directory):
        print(f"Erreur: Le répertoire {args.directory} n'existe pas")
        return
    
    print("Début du traitement...")
    total_files, total_chunks = process_directory(
        args.directory,
        args.persist_dir
    )
    
    print("\nRésumé du traitement:")
    print(f"Nombre total de fichiers traités: {total_files}")
    print(f"Nombre total de chunks créés: {total_chunks}")
    print(f"Base de données stockée dans: {args.persist_dir}")

if __name__ == "__main__":
    main()

# Pour exécuter le script, vous pouvez utiliser la commande suivante :
# python load_chroma.py <chemin_vers_le_répertoire> --persist_dir <répertoire_de_persistance>