from class_scrapping import UQACScraper
# Créer une instance du scraper
scraper = UQACScraper()

# Démarrer le scraping
documents = scraper.start_scraping()

# Sauvegarder les résultats
scraper.save_to_file("donnees_manuel_gestion.txt")