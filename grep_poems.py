"""
    Download Victor Hugo's poems from https://poesie.webnet.fr/
"""

import requests
from bs4 import BeautifulSoup

def get_list():
    urls = []
    document = BeautifulSoup(requests.get("https://poesie.webnet.fr/lesgrandsclassiques/Poemes/victor_hugo").content)
    elements = document.select('.author-list__link')
    for element in elements:
        urls.append(element['href'])
    return urls

def get_poem(url):
    document = BeautifulSoup(requests.get('https://poesie.webnet.fr/' + url).content, "html.parser")
    return document.select('.poem__content')[0].get_text(separator="\n")

urls = get_list()
result = ""
for url in urls:
    result += get_poem(url)

with open("victorhugo.txt", "w+") as file:
    file.write(result)
