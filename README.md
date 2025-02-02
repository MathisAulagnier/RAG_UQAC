# RAG_UQAC
Implémenter un chatbot utilisant la technique de Retrieval Augmented Generation (RAG) à l’aide de la librairie Python Langchain. L’objectif principal est de créer un système capable de répondre à des questions en s’appuyant sur des données extraites d’un manuel de gestion disponible sur le site de l’UQAC.

## Scraping des Pages Web en Markdown

### Objectif

Le but de cette partie est de scraper des pages web pour en extraire le contenu sous forme de fichiers Markdown. Les URLs des pages à traiter sont récupérées depuis un sitemap, puis le contenu est extrait, nettoyé et enregistré.

### Méthodes Utilisées

1. **Récupération des URLs** :
    - Les URLs sont extraites de manière récursive à partir d’un sitemap XML. Les sitemaps imbriqués sont également pris en compte.
2. **Extraction et Conversion du Contenu** :
    - Chaque page web est scrappée pour en extraire le contenu pertinent, puis ce contenu est converti en format Markdown.
3. **Enregistrement des Résultats** :
    - Les contenus extraits sont sauvegardés dans des fichiers Markdown, organisés dans un dossier spécifié par l’utilisateur.

### Programmes

#### getURL.py

- **Objectif** : Récupérer toutes les URLs à partir d’un sitemap XML et de ses sitemaps imbriqués de manière récursive.

#### getArticle.py

- **Objectif** : Extraire le contenu d’un article d’une page web et le sauvegarder en Markdown. Le contenu est nettoyé pour ne garder que l’article pertinent, et le fichier est enregistré dans un dossier spécifié au format Markdown.

#### getUQAC_Gestion.py

- **Objectif** : Exécuter le scraping des articles du site de l’UQAC en utilisant les fonctions de `getURL.py` et `getArticle.py`.

### Exécution

Pour exécuter le processus de scraping et d’enregistrement des articles en Markdown, lancez le script suivant avec l’option `--output` pour spécifier le dossier de sortie :

```bash
cd src/scrapping
python scrapping.py --output <nom_dossier>
```

## Base de Données Vectorielle et Embedding

### Objectif

Cette partie vise à transformer les documents Markdown en une base de données vectorielle permettant une recherche sémantique efficace. Le processus comprend la création de chunks (segments de texte), leur embedding (transformation en vecteurs), et leur stockage dans une base de données vectorielle.

### Méthodes Utilisées

1. **Découpage en Chunks** :
   - Les documents Markdown sont découpés en segments plus petits pour un traitement optimal
   - Utilisation de `MarkdownTextSplitter` avec overlap pour maintenir le contexte
   - Extraction de l'URL source depuis le header de chaque document

2. **Embedding des Chunks** :
   - Transformation des chunks en vecteurs via le modèle Ollama
   - Conservation des métadonnées (URL source) pour chaque chunk
   - Dimension des vecteurs : 4096 (modèle llama2)

3. **Stockage Vectoriel** :
   - Utilisation de Chroma DB comme base de données vectorielle
   - Persistance locale des données pour réutilisation
   - Association des métadonnées (source) à chaque chunk

### Programmes

#### create_chunk.py

- **Objectif** : Découper les documents Markdown en chunks cohérents avec overlap
- **Fonctionnalités** :
  - Lecture des fichiers Markdown
  - Découpage en segments avec préservation du contexte
  - Gestion de la taille des chunks et de l'overlap

#### save_chunk.py

- **Objectif** : Stocker les chunks dans la base de données vectorielle Chroma
- **Fonctionnalités** :
  - Extraction de l'URL source
  - Génération des embeddings
  - Stockage des chunks avec leurs métadonnées

#### process_markdown_files.py

- **Objectif** : Automatiser le traitement de multiples fichiers Markdown
- **Fonctionnalités** :
  - Traitement en lot des fichiers d'un dossier
  - Réinitialisation de la base de données
  - Génération de statistiques de traitement

### Exécution

Pour traiter les fichiers Markdown et créer la base de données vectorielle :

```bash
cd src/
python load_chroma.py scrapping/output_UQAC_Website/
```

### Structure des Données

- **Format des Chunks** :
  - Premier chunk : URL source (#https://example.com)
  - Chunks suivants : Contenu avec overlap
  - Métadonnées : URL source associée
- **Base de Données** :
  - Type : Chroma DB
  - Stockage : Local (./chroma_db par défaut)
  - Format : Vecteurs d'embedding + métadonnées
 
### Application

- **Lancé le Serveur Ollama en Local** :
  Ouvrir un terminal Windows PowerShell
  puis taper ces deux commandes à la suite :  
    ```bash
    $env:OLLAMA_HOST="0.0.0.0"
    ollama serve
    ```
- **Lancé l'Application** :
  Ouvrir un deuxième terminal Windows PowerShell
  puis taper ces deux commandes à la suite (en remplacant le chemin de la commande cd par votre propre chemin vers de dossier "App" du projet):  
    ```bash
    cd C:\Users\ryan4\Atelier_pratique2\Projet1\src\App
    streamlit run app_version_gpt.py
    ```





A faire : 
 -  Tester une nouvelle version d'embedding 
 -  Summary des conversation pour l'historique afin de ne pas surcharger la mémoire (Limite de Token)
 Présenter dans le rappmort les différents tests de modele et les justifier:
 
  -  Modèles pour l'embedding : all-MiniLM-L6-v2 / Llama3 / Nomic-embedding
  -  Modèles pour la génération : GPT-2 / Mistall-small / Llama3 / DeepSeek 8B / Mistral 7B double appel
