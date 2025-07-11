import requests
from bs4 import BeautifulSoup
import json
import os,re,time

output_dir = "data/raw/climate_gov"

url = "https://www.climate.gov/sitemap.xml?page=2"
resp = requests.get(url)
soup_1 = BeautifulSoup(resp.content, "xml")

article_urls = [loc.text for loc in soup_1.find_all("loc")]

def clean_filenames(title: str) -> str:
  filename = re.sub(r'[<>:"/\\|?*]', "_", title)
  return filename[:100].strip()

def scrape_article(url):
  try:
    res = requests.get(url, timeout=10)
    res.raise_for_status
    soup = BeautifulSoup(res.content, "html.parser")

    title = soup.find("h1").text.strip() if soup.find("h1") else "untitled"
    date_tag = soup.find("time")
    date = date_tag.get("datetime") if date_tag and date_tag.has_attr("datetime") else None
    paragraphs = soup.find_all("p")
    content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

    return {
      "url": url,
      "title": title,
      "date": date,
      "content": content,
      "source": "climate.gov"
    }
  except Exception as e:
    print(f"Error scraping {url}: {e}")
    return None

for i, url in enumerate(article_urls):
  if not url.startswith("http") or any(url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".pdf", ".zip"]):
    print(f"Skipping non-article URL: {url}")
    continue

  print(f"[{i+1}/{len(article_urls)}] Downloading: {url}")

  article_data = scrape_article(url)
  if article_data:
    filename = f"{i+2001:04d}_{clean_filenames(article_data['title'])}.json"
    file_path = output_dir + "/" + filename

    if os.path.exists(file_path):
      print(f"Skipping existing file: {filename}")
      continue

    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(article_data, f, indent=2, ensure_ascii=False)

  time.sleep(1)