import asyncio
import argparse
from pathlib import Path
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

async def extract_article_to_markdown(url: str, output_dir: str = "output") -> None:
    """
    Extrait le contenu d'un article d'une page web et le sauvegarde en markdown.
    
    Args:
        url (str): L'URL de la page à traiter
        output_dir (str): Le dossier où sauvegarder le fichier markdown
    """
    # Créer le dossier de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Configuration du navigateur
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True
    )
    
    # Configuration du générateur markdown
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.4,
            threshold_type="fixed"
        )
    )
    
    # Configuration du crawler avec un script JS pour isoler l'article
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        js_code=["""
            (() => {
                // Supprimer tout sauf l'article
                const article = document.querySelector('article[id^="post-"]');
                if (!article) return;
                
                // Nettoyer le contenu non désiré
                const sidebar = document.querySelector('#content-sidebar');
                if (sidebar) sidebar.remove();
                
                // Créer un nouveau body avec uniquement l'article
                document.body.innerHTML = '';
                document.body.appendChild(article);
            })();
        """]
    )
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            print(f"Traitement de l'URL: {url}")
            result = await crawler.arun(url, config=crawler_config)
            
            # Récupérer le contenu markdown selon la structure retournée
            markdown_content = result.markdown
            if hasattr(markdown_content, 'fit_markdown'):
                markdown_content = markdown_content.fit_markdown
            
            if markdown_content:
                # Créer un nom de fichier sécurisé basé sur l'URL
                safe_filename = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
                if not safe_filename:
                    safe_filename = 'index'
                safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in ('-', '_'))
                filename = f"{safe_filename}.md"
                filepath = output_path / filename
                
                # Ajouter l'URL comme titre au début du markdown
                content = f"# {url}\n\n{markdown_content}"
                
                # Sauvegarder le contenu
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"Contenu sauvegardé dans: {filepath}")
            else:
                print("Aucun contenu d'article trouvé dans la page.")
                
    except Exception as e:
        print(f"Erreur lors du traitement de {url}: {str(e)}")
        raise  # Pour voir la stack trace complète en mode debug

def main():
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Extrait le contenu d'un article web en markdown")
    parser.add_argument('url', help='L\'URL de la page à traiter')
    parser.add_argument('--output', '-o', default='output',
                      help='Le dossier où sauvegarder le fichier markdown (défaut: output)')
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Exécuter l'extraction
    asyncio.run(extract_article_to_markdown(args.url, args.output))

if __name__ == "__main__":
    main()