import subprocess
import asyncio
import argparse

from getURL import get_recursive_urls

# Fonction pour scraper et enregistrer chaque URL dans un dossier donné
async def scrape_and_save_urls(urls: list, output_dir: str):
    for url in urls:
        print(f"Scraping {url}...")
        # Utilisation de subprocess pour exécuter le script.py pour chaque URL
        # Vous pouvez passer l'URL et le dossier de sortie en argument
        await run_scraper(url, output_dir)

# Fonction qui appelle le script de scraping
async def run_scraper(url: str, output_dir: str):
    command = [
        'python', 'getArticle.py', url, '--output', output_dir
    ]
    
    # Utilisation de subprocess pour exécuter le script.py
    process = await asyncio.create_subprocess_exec(*command)
    await process.communicate()

def main():
    # Argument parser pour le dossier de sortie
    parser = argparse.ArgumentParser(description="Scraper des pages web et les enregistrer en markdown")
    parser.add_argument('--output', '-o', default='output', help='Le dossier où sauvegarder les fichiers markdown')
    args = parser.parse_args()

    # Récupération des URLs à scraper
    sitemap_url = "https://www.uqac.ca/mgestion/wp-sitemap.xml"
    print(f"Fetching URLs from {sitemap_url}")
    urls = get_recursive_urls(sitemap_url)
    
    # Lancer le processus de scraping pour toutes les URLs
    asyncio.run(scrape_and_save_urls(urls, args.output))

if __name__ == "__main__":
    main()