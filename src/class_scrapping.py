import requests
from bs4 import BeautifulSoup
import tempfile
import os
from urllib.parse import urljoin
import PyPDF2
import logging
from typing import List, Dict


class UQACScraper:
    def __init__(self, base_url: str = "https://www.uqac.ca/mgestion/"):
        self.base_url = base_url
        self.visited_urls = set()
        self.documents = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_html_content(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extrait le contenu des balises spécifiques d'une page HTML."""
        content = {
            "header": "",
            "content": "",
            "url": ""
        }
        
        # Extraction du header
        header = soup.find("div", class_="entry-header")
        if header:
            content["header"] = header.get_text(strip=True)
            
        # Extraction du contenu principal
        main_content = soup.find("div", class_="entry-content")
        if main_content:
            content["content"] = main_content.get_text(strip=True)
            
        return content

    def extract_pdf_content(self, pdf_url: str) -> str:
        """Télécharge et extrait le contenu d'un fichier PDF."""
        try:
            # Création d'un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                # Téléchargement du PDF
                response = requests.get(pdf_url)
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            # Extraction du texte du PDF
            text = ""
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Suppression du fichier temporaire
            os.unlink(temp_file_path)
            return text

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction du PDF {pdf_url}: {str(e)}")
            return ""

    def get_links(self, url: str) -> List[str]:
        """Récupère tous les liens d'une page."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if full_url.startswith(self.base_url) and full_url not in self.visited_urls:
                    links.append(full_url)
                    
            return links
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des liens de {url}: {str(e)}")
            return []

    def scrape_page(self, url: str) -> Dict[str, str]:
        """Scrape une page spécifique (HTML ou PDF)."""
        try:
            if url in self.visited_urls:
                return {}
                
            self.visited_urls.add(url)
            self.logger.info(f"Traitement de l'URL: {url}")

            # Vérification si c'est un PDF
            if url.lower().endswith('.pdf'):
                content = self.extract_pdf_content(url)
                return {
                    "content": content,
                    "url": url,
                    "type": "pdf"
                }
            else:
                # Traitement des pages HTML
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                content = self.extract_html_content(soup)
                content["url"] = url
                content["type"] = "html"
                return content

        except Exception as e:
            self.logger.error(f"Erreur lors du scraping de {url}: {str(e)}")
            return {}

    def start_scraping(self):
        """Démarre le processus de scraping."""
        urls_to_visit = [self.base_url]
        
        while urls_to_visit:
            current_url = urls_to_visit.pop(0)
            
            # Scrape la page courante
            content = self.scrape_page(current_url)
            if content:
                self.documents.append(content)
            
            # Récupère les nouveaux liens
            new_links = self.get_links(current_url)
            urls_to_visit.extend([url for url in new_links if url not in self.visited_urls])

        self.logger.info(f"Scraping terminé. {len(self.documents)} documents traités.")
        return self.documents

    def save_to_file(self, filename: str = "scraped_data.txt"):
        """Sauvegarde les données scrapées dans un fichier."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for doc in self.documents:
                    f.write(f"URL: {doc['url']}\n")
                    f.write(f"Type: {doc['type']}\n")
                    if doc['type'] == 'html':
                        f.write(f"Header: {doc['header']}\n")
                    f.write(f"Content: {doc['content']}\n")
                    f.write("-" * 80 + "\n")
                    
            self.logger.info(f"Données sauvegardées dans {filename}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")