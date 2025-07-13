import os
import json
import asyncio
from tqdm import tqdm
from openai import OpenAI
from functools import partial
import random

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

MAX_TOKENS_BUDGET = 12_500_000  # ~ $5 budget @ 0.0004 per 1k tokens
tokens_used = 0
tokens_lock = asyncio.Lock()

MAX_CONCURRENT_REQUESTS = 5  # limit concurrency
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def count_tokens(messages):
  text = "".join(m["content"] for m in messages)
  return int(len(text.split()) * 1.3)

async def generate_one(article_text, instruction, max_tokens=512, retries=3):
  global tokens_used

  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{instruction}\n\nArticle:\n{article_text[:1000]}"}
  ]

  prompt_tokens = count_tokens(messages)

  async with semaphore:
    for attempt in range(retries):
      async with tokens_lock:
        if tokens_used + prompt_tokens + max_tokens > MAX_TOKENS_BUDGET:
          print("Reached token budget limit, stopping further requests.")
          return None
      try:
        loop = asyncio.get_running_loop()
        # Wrap sync call in executor
        response = await loop.run_in_executor(
          None,
          partial(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
          )
        )
        usage = response.usage
        total_tokens = usage.total_tokens

        async with tokens_lock:
          tokens_used += total_tokens

        output = response.choices[0].message.content.strip()
        return {
          "instruction": instruction,
          "input": article_text[:1000],
          "output": output
        }
      except Exception as e:
        wait_time = 2 ** attempt + random.uniform(0, 1)
        print(f"Error on attempt {attempt+1}: {e}. Retrying in {wait_time:.1f}s...")
        await asyncio.sleep(wait_time)

    print(f"Failed after {retries} retries for instruction: {instruction}")
    return None

async def generate_instruction_pairs(article_text, num_variants=3):
  instructions = [instruction_templates[i % len(instruction_templates)] for i in range(num_variants)]
  tasks = [generate_one(article_text, instr) for instr in instructions]
  results = await asyncio.gather(*tasks)
  return [r for r in results if r]

async def main():
  processed_dir = "data/processed"
  output_dir = "src/final_dataset"
  os.makedirs(output_dir, exist_ok=True)

  final_dataset = []

  for file in os.listdir(processed_dir):
    with open(os.path.join(processed_dir, file), "r", encoding="utf-8") as f:
      articles = json.load(f)

    for article in tqdm(articles, desc=f"Generating from {file}"):
      content = article.get("content", "")
      if len(content) < 300:
        continue
      pairs = await generate_instruction_pairs(content)
      if not pairs:
        print("Stopping generation due to token budget or errors.")
        break
      final_dataset.extend(pairs)

  out_path = os.path.join(output_dir, "gpt3.5_turbo_instruction_data.json")
  with open(out_path, "w", encoding="utf-8") as out:
    json.dump(final_dataset, out, indent=2, ensure_ascii=False)

  print(f"✅ Done! {len(final_dataset)} instruction pairs saved to {out_path}")

if __name__ == "__main__":
  asyncio.run(main())
