import os
import json

RAW_DIR = "data/raw/climate_gov"  # adjust as needed

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".json"):
        filepath = os.path.join(RAW_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                article = json.load(f)

                article["source"] = "climate.gov"

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f" Error processing {filename}: {e}")
