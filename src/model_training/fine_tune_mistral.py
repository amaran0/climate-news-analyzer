import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

#CONFIG
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "./src/final_dataset"
OUPUT_DIR = ""