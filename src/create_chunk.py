from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def read_markdown_file(file_path):
    """Lit un fichier markdown et retourne son contenu"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_markdown_into_chunks(markdown_text, chunk_size=500, chunk_overlap=25):
    """
    Découpe le texte markdown en chunks avec overlap
    
    Args:
        markdown_text (str): Le texte markdown à découper
        chunk_size (int): La taille maximale de chaque chunk
        chunk_overlap (int): Le nombre de caractères qui se chevauchent entre les chunks
    
    Returns:
        list: Liste des chunks
    """
    # Création du splitter spécifique pour Markdown
    markdown_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Alternative avec RecursiveCharacterTextSplitter
    # markdown_splitter = RecursiveCharacterTextSplitter(
    #     separators=["\n\n", "\n", " ", ""],
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function=len
    # )

    # Découpage du texte en chunks
    chunks = markdown_splitter.split_text(markdown_text)
    
    return chunks

def main():
    # Chemin vers votre fichier markdown
    file_path = "scrapping/output_UQAC_Website/annexe-4-procedure-pour-le-partage-de-linformation-aux-fonds-de-recherche-du-quebec.md"
    
    # Lecture du fichier
    markdown_text = read_markdown_file(file_path)
    
    # Découpage en chunks
    chunks = split_markdown_into_chunks(
        markdown_text,
        chunk_size=1000, 
        chunk_overlap=100  
    )
    
    # Affichage des résultats
    print(f"Nombre total de chunks : {len(chunks)}")
    
    # Affichage des premiers chunks pour vérification
    for i, chunk in enumerate(chunks[:5]):  # Affiche les 3 premiers chunks
        print(f"\nChunk {i+1}:")
        print("-" * 50)
        print(chunk)
        print("-" * 50)

if __name__ == "__main__":
    main()
