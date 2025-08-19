import json
import random

# Paths
TRAIN_JSON = "./src/final_dataset/gpt3.5_turbo_instruction_data.json"
BENCHMARK_JSON = "./src/testing_dataset/test_dataset.json"
TEST_OUTPUT = "./src/testing_dataset/test_dataset_600.json"    # final test set (100 + 500)

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
    benchmark_100 = json.load(f)

random.seed(42)
sample_500 = random.sample(train_data, 500)

final_test_set = benchmark_100 + sample_500

with open(TEST_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(final_test_set, f, indent=2, ensure_ascii=False)

print(f"Combined test set saved to {TEST_OUTPUT} with {len(final_test_set)} examples")
