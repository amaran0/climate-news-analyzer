from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_DIR = "./src/mistral-qlora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
  BASE_MODEL,
  torch_dtype=torch.float16,
  device_map="auto"
)

ft_model = AutoModelForCausalLM.from_pretrained(
  BASE_MODEL,
  torch_dtype=torch.float16,
  device_map="auto"

)
ft_model = PeftModel.from_pretrained(ft_model, LORA_DIR)

def generate_answer(model, tokenizer, instruction, inp, max_new_tokens=256):
  prompt = f"Instruction: {instruction}\nInput: {inp}\nAnswer:"
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  with torch.no_grad():
    output_ids = model.generate(
      **inputs,
      max_new_tokens = max_new_tokens,
      do_sample=False,
      temperature=0.7,
      top_p=0.9
    )
  return tokenizer.decode(output_ids[0], skip_special_tokens=True)

