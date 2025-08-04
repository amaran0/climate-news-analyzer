import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login

#CONFIG
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "./src/final_dataset/gpt3.5_turbo_instruction_data.json"
OUTPUT_DIR = "./src/mistral-qlora"
login(token=os.getenv("HUGGINGFACE_TOKEN"))

def format_example(example):
  if example["input"]:
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
  else:
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
  
def tokenize(example, tokenizer):
  prompt = format_example(example)
  tokenized = tokenizer(
    prompt,
    truncation=True,
    max_length=2048,
    padding="max_length"
  )
  return tokenized

print("🤞Loading dataset")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

print("🪙Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer))

print("Loading base model")
model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  load_in_4bit=True,
  torch_dtype="auto",
  device_map="auto"
)

print("😊Preparing model for QLORA training")
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
  r=64,
  lora_alpha=16,
  target_modules=["q_proj","v_proj","k_proj","o_proj"],
  lora_dropout=0.05,
  bias="none",
  task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

print("👌Training!!!")
args = TrainingArguments(
  OUTPUT_DIR=OUTPUT_DIR,
  per_device_train_batch_size=2,
  gradient_accumulation_steps=4,
  num_train_epochs=3,
  logging_steps=10,
  save_steps=200,
  save_total_limit=2,
  fp16=False,
  bf16=True,
  report_to="none"
)

trainer = Trainer(
  model=model,
  train_dataset=tokenized_dataset,
  args=args,
  tokenizer=tokenizer
)

trainer.train()

print(f"Saving QLORA weights to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)