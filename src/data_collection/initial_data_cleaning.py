import os, json, re, html
from pathlib import Path

input_dir = Path("data/raw")
output_dir = Path("data/processed")

def clean_text(text):
  text = html.unescape(text)
  text = text.replace("\\n", "\n").replace("\\", "").replace("  ", " ")

  boilerplate_patterns = [
    r"Subscribe to (our )?newsletter.*",
    r"Follow (us|NASA|NOAA) on.*",
    r"©?\s?\d{4}?.*",
    r"Privacy Policy",
    r"Terms of Use",
    r"Donate.*",
    r"Last updated.*",
    r"Accessibility Statement",
    r"Contact Us",
    r"NOAA Climate\.gov",
    r"NASA.*Official.*",
    r"FOIA|No Fear Act|Inspector General"
  ]
  for pattern in boilerplate_patterns:
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

  text = re.sub(r"\n{2,}", "\n\n", text)
  text = text.strip()
  return text

def clean_and_store_articles(input, output, min_len=300):
  seen_contents = set()

  for source in os.listdir(input):
    source_path = os.path.join(input, source)
    if not os.path.isdir(source_path):
      continue

    all_cleaned_articles = []

    for file in os.listdir(source_path):
      file_path = os.path.join(source_path, file)
      with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      if isinstance(data, dict):
        data = [data]

      for article in data:
        content = clean_text(article.get("content", ""))
        if len(content) < min_len or content in seen_contents:
          continue
        seen_contents.add(content)
        article["content"] = content
        all_cleaned_articles.append(article)

    output_file = os.path.join(output_dir, f"{source}.json")
    with open(output_file, "w", encoding="utf-8") as out_f:
      json.dump(all_cleaned_articles, out_f, indent=2, ensure_ascii=False)

clean_and_store_articles(input_dir, output_dir)
