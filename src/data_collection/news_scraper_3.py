import requests
from bs4 import BeautifulSoup
import json
import os,re,time

output_dir = "data/raw/noaa_gov"

url = "https://www.noaa.gov/sitemap.xml?page=1"
resp = requests.get(url)
soup_1 = BeautifulSoup(resp.content, "xml")

article_urls = [loc.text for loc in soup_1.find_all("loc")]

def clean_filenames(title: str) -> str:
  filename = re.sub(r'[<>:"/\\|?*]', "_", title)
  return filename[:100].strip()

def scrape_article(url):
  try:
    res = requests.get(url, timeout=10)
    if (res.status_code!= 200):
      print(f"❌ Skipping {url}: got {res.status_code}")
    res.raise_for_status
    soup = BeautifulSoup(res.content, "html.parser")

    title_tag = soup.find("div", property="og:title")
    title = title_tag["content"].strip() if title_tag else soup.title.string.strip()
    date_tag = soup.find("time")
    date = date_tag.get("datetime") if date_tag and date_tag.has_attr("datetime") else None
    content_area = soup.find("div", class_="c-field__content")
    if content_area:
      paragraphs = content_area.findAll("p")
      content = "\n".join(p.get_text(strip=True) for p in paragraphs)
    else:
      content = ""

    return {
      "url": url,
      "title": title,
      "date": date,
      "content": content,
      "source": "noaa.gov"
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
    filename = f"{i+1:04d}_{clean_filenames(article_data['title'])}.json"
    file_path = output_dir + "/" + filename

    if os.path.exists(file_path):
      print(f"Skipping existing file: {filename}")
      continue

    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(article_data, f, indent=2, ensure_ascii=False)

  time.sleep(1)