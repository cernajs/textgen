
page = "https://www.fridayflashfiction.com/100-word-stories/previous/"
output_file = "blog_data.txt"
writer = open(output_file, 'a')

import requests
from bs4 import BeautifulSoup
import re

def request_page(page):
    response = requests.get(page)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        blog_divs = soup.find_all('div', id=re.compile(r'^blog-post-\d+'))

        for blog_div in blog_divs:
            title = blog_div.find('h2').text.strip() if blog_div.find('h2') else "No Title"

            content = blog_div.find('div', class_='blog-content').text.strip() if blog_div.find('div', class_='blog-content') else "No Content"

            name = title.split("by")[0][:-2]

            combined = f"<|title|>\n{name}\n<|story|>\n{content}\n"
            writer.write(f"{combined}\n")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


for i in range(1, 101):
    request_page(page + str(i))
