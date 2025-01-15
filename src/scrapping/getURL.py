import requests
from xml.etree import ElementTree

def get_recursive_urls(sitemap_url):
    """
    Fetches all URLs recursively from a sitemap and its nested sitemaps.

    Args:
        sitemap_url (str): The URL of the sitemap to process.

    Returns:
        List[str]: List of URLs
    """
    urls = []

    try:
        # Fetch the sitemap XML
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML content
        root = ElementTree.fromstring(response.content)

        # Define the namespace
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        # Find all loc elements, which contain URLs
        locs = root.findall('.//ns:loc', namespace)
        
        for loc in locs:
            url = loc.text
            if url.endswith('.xml'):
                # If the URL points to another sitemap, fetch and recurse
                print(f"Found nested sitemap: {url}")
                urls.extend(get_recursive_urls(url))
            else:
                # Otherwise, it's a regular URL, so add it to the list
                urls.append(url)

    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")

    return urls

if __name__ == "__main__":
    sitemap_url = "https://www.uqac.ca/mgestion/wp-sitemap.xml"
    print(f"Fetching URLs from {sitemap_url}")
    urls = get_recursive_urls(sitemap_url)
    # Ecrire les URLs dans un fichier
    with open('urls.txt', 'w') as f:
        for url in urls:
            f.write(url + '\n')
            
    print(f"Found {len(urls)} URLs")
