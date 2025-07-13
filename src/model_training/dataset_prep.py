import os, json, time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.getenv("GROQ_API_KEY")
)

instruction_templates = [
  "Summarize this article in one paragraph.",
  "Write a brief overview of the article in 3–5 sentences.",
  "List 3 key takeaways from this article.",
  "What are the main findings in this article?",
  "Generate 3 question-answer pairs based on this article.",
  "Write a quiz with 3 short questions and answers based on the article.",
  "How might the findings in this article affect climate policy?",
  "Explain this article in simple terms for a high school student.",
  "Imagine you're teaching a class. What would you say about this article?"
]

def generate_instruction_pairs(article_text, num_variants=9):
  results = []
  for i in range(num_variants):
    instruction = instruction_templates[i % len(instruction_templates)]
    try:
      response = client.chat.completions.create(
        model="mixtral-8x7b-instruct",
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f"{instruction}\n\nArticle:\n{article_text}"}
        ],
        temperature=0.7,
        max_tokens=512
      )
      output = response.choices[0].message.content.strip()
      results.append({
        "instruction": instruction,
        "input": article_text[:1000],
        "output": output
      })
    except Exception as e:
      print("Error:", e)
      time.sleep(2)
  return results

processed_dir = "data/processed"
output_dir = "src/final_dataset"
final_dataset = []

for file in os.listdir(processed_dir):
  with open(os.path.join(processed_dir, file), "r", encoding="utf-8") as f:
    articles = json.load(f)

  for article in tqdm(articles, desc=f"Generating from {file}"):
    content = article.get("content", "")
    if len(content) < 300:
      continue

    pairs = generate_instruction_pairs(content)
    final_dataset.extend(pairs)

with open(os.path.join(output_dir, "mistral_instruction_data.json"), "w", encoding="utf-8") as out:
  json.dump(final_dataset, out, indent=2, ensure_ascii=False)

print(f"✅ Done! {len(final_dataset)} instruction pairs saved.")