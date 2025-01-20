from chromadb.config import Settings
from chromadb import Client
from langchain.embeddings import OllamaEmbeddings

# Charger la base vectorielle
client = Client(Settings(persist_directory="../chroma_db/chroma.sqlite3"))
collection = client.get_collection("UQAC_documents")

# Générer un embedding pour le prompt utilisateur
embedder = OllamaEmbeddings(model="llama2")
user_prompt = "Comment gérer un projet avec des ressources universitaire de type autobiographique ?"  # Prompt random
user_embedding = embedder.embed_query(user_prompt)

# Recherche dans la base vectorielle
results = collection.query(
    query_embeddings=[user_embedding],
    n_results=5,
)

# Affichage des résultats
for idx, (document, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
    print(f"Résultat {idx + 1}:")
    print(f"Contenu : {document}")
    print(f"Métadonnées : {metadata}")
