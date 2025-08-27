from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from sentence_transformers import SentenceTransformer
from load_models import base_model, ft_model, tokenizer, generate_answer
import json

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate(models, dataset, tokenizer):
  results = {name: {"rouge1": [], "rougeL": [], "cosine": [], "latency": []} for name in models.keys()}
  
  for example in tqdm(dataset, desc="Evaluating"):
    instr, inp, out = example["Instruction"], example["input"], example["output"]
    
    for name, model in model.items():
      import time
      start = time.time()
      pred = generate_answer(model, tokenizer, instr, inp)
      end = time.time()
      
      # ROUGE
      scores = scorer.score(out, pred)
      results[name]["rouge1"].append(scores["rouge1"].fmeasure)
      results[name]["rougeL"].append(scores["rougeL"].fmeasure)
      
      # Embedding similarity
      out_emb = embedder.encode([out])
      pred_emb = embedder.encode([pred])
      sim = cosine_similarity(out_emb, pred_emb)[0][0]
      results[name]["cosine"].append(sim)
      
      # Latency
      results[name]["latency"].append(end - start)
      
  return results

with open("./src/testing_dataset/test_dataset_600.json") as f:
  dataset = json.load(f)
  
models = {"base": base_model, "fine-tuned": ft_model}
results = evaluate(models, dataset, tokenizer)