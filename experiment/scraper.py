import goose3
from pprint import pprint
from magic_google import MagicGoogle

PROXIES = None
def get_google_urls(query):
    mg = MagicGoogle(PROXIES)
    result = mg.search_page(query=query)
    print(result)
    for url in mg.search_url(query=query):
        pprint(url)
    for item in mg.search(query=query, num=1):
        pprint(item)

if __name__ == "__main__":
    get_google_urls('physical medicine')
