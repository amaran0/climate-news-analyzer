import numpy as np
import matplotlib.pyplot as plt
import json

def summarize_results(results):
  summary = {}
  for model, metrics in results.items():
    summary[model] = {m: np.mean(vals) for m, vals in metrics.items()}
  return summary

results = ""
with open("results/eval_results.json") as f:
  results = json.load(f)

summary = summarize_results(results)
print(summary)

metrics = ["rouge1", "rouge:L", "cosine", "latency"]
for m in metrics:
  plt.bar(summary.keys(), [summary[model][m] for model in summary], label=m)
  plt.title(m)
  plt.ylabel(m)
  plt.show()