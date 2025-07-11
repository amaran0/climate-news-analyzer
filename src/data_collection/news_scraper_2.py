import os,re,time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

url = "https://www.climate.gov/sitemap.xml"
output_dir = "data/raw/climate_gov"

def clean_filename(title: str) -> str:
  filename = re.sub(r'[<>:"/\\|?*]', '_', title)
  return filename[:100].strip()

def is_html_url(url: str) -> bool:
  non_html_exts = [".jpg", ".jpeg", ".png", ".pdf", ".zip", ".gif", ".svg", ".mp4"]
  return not any(url.lower().endswith(ext) for ext in non_html_exts)

def scrape_article(url: str) -> dict | None:
  try:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    title = soup.find("h1")
    title_text = title.text.strip() if title else "untitled"

    date_tag = soup.find("time")
    date = date_tag.get("datetime") if date_tag and date_tag.has_attr("datetime") else None

    paragraphs = soup.find_all("p")
    content = "\n".join(p.text.strip() for p in paragraphs if p.text.strip())

    return {
      "url": url,
      "title": title_text,
      "date": date,
      "content": content,
      "source": "climate.gov"
    }
  except Exception as e:
    print(f"Error scraping {url}: {e}")
    return None

def main():
  sitemap_resp = requests.get(url)
  sitemap_resp.raise_for_status()
  sitemap_soup = BeautifulSoup(sitemap_resp.content, "xml")
  urls = [loc.text for loc in sitemap_soup.find_all("loc")]

  for i, url in enumerate(urls):
    if not is_html_url(url):
      print(f"Skipping non-HTML URL: {url}")
      continue

    print(f"[{i+1}/{len(urls)}] Scraping: {url}")
    article = scrape_article(url)
    if article:
      filename = f"{i+1:04d}_{clean_filename(article['title'])}.json"
      file_path = os.path.join(output_dir, filename)

      if os.path.exists(file_path):
        print(f"Skipping existing file: {filename}")
        continue

      with open(file_path, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)

    time.sleep(1)

if __name__ == "__main__":
    main()
