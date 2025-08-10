import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login

#CONFIG
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "./src/final_dataset/gpt3.5_turbo_instruction_data.json"
OUTPUT_DIR = "/content/drive/MyDrive/mistral-qlora"
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
  tokenized["labels"] = tokenized["input_ids"].copy()
  tokenized["labels"] = [ (tok if tok != tokenizer.pad_token_id else -100) for tok in tokenized["labels"] ]
  return tokenized

print("🤞Loading dataset")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print("🪙Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

tokenized_train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer))
tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize(x, tokenizer))


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
model.config.use_cache = False

print("👌Training!!!")
args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  per_device_train_batch_size=2,
  gradient_accumulation_steps=4,
  num_train_epochs=3,
  logging_steps=10,
  save_steps=200,
  eval_strategy="steps",
  eval_steps=200,
  save_total_limit=2,
  fp16=False,
  bf16=True,
  gradient_checkpointing=True,
  report_to="none",
  load_best_model_at_end=True,
  metric_for_best_model="loss"
)

trainer = Trainer(
  model=model,
  train_dataset=tokenized_train_dataset,
  eval_dataset=tokenized_eval_dataset,
  args=args,
  tokenizer=tokenizer,
  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

)

trainer.train()

print(f"Saving QLORA weights to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)