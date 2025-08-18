import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

#CONFIG
CHROMA_DIR = "./src/chroma_db"
BASE_MODEL = "mistralai/Mistral-7b-v0.1"
LORA_DIR = "./src/mistral-qlora"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#load retriever
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_fn)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
retriever = retriever.with_retry(stop_after_attempt=1)

#load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
  BASE_MODEL,
  load_in_4bit=True,
  device_map="auto",
  torch_dtype="auto"
)
model = PeftModel.from_pretrained(model, LORA_DIR)
model.eval()

OFF_TOPIC_KEYWORDS = ["hello", "hi", "how are you", "good morning", "thank you"]

def generate_answer(query: str):
  if any(word in query.lower() for word in OFF_TOPIC_KEYWORDS):
    return "Hello! I am ClimateAI, an expert in climate science. Please ask a climate-related question."
  
  docs = retriever.invoke(query)
  if not docs:
    context = "None"
  for i, doc in enumerate(docs, 1):
    url = doc.metadata.get("source", "Unknown source")
    context += f"[Source {i}] {url}\n{doc.page_content}\n\n"

  prompt = f"""### Instruction:
  You are ClimateAI, an AI assistant specialized in climate science, climate policy, and environmental research.
  - Always provide answers that are grounded in the context provided.
  - When referencing facts, cite the sources using the [Source #] notation provided.
  - If the question is unrelated to climate or environment, politely respond that you can only answer climate-related questions, then end your answer.
  - Your answers should be detailed and thorough, explaining reasoning step by step when applicable.
  - Avoid hallucinations; do not invent facts.

  ### Context:
  {context}

  ### Question:
  {query}

  ### Response:
  """

  inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=2048
  ).to(DEVICE)

  outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.4,
    top_p=0.6,
    pad_token_id=tokenizer.eos_token_id
  )
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)

  return response.split("### Response:")[-1].strip()

print("RAG query ready. Ask a question:")
while True:
  try:
    query = input("You > ").strip()
    if not query:
      continue
    print("ClimateAI > Thinking...\n")
    print(generate_answer(query))
    print()
  except KeyboardInterrupt:
    print("\nExiting.")
    break