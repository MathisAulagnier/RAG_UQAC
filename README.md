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
