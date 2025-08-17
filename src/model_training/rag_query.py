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

def generate_answer(query: str):
  docs = retriever.get_relevant_documents(query)
  context = "\n\n".join([doc.page_content for doc in docs])

  prompt = f"""### Instruction:
  Use the context below to answer the question.

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
  outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)

  return response.split("### Response:")[-1].strip()

print("RAG query ready. Ask a question:")
while True:
  try:
    query = input("You > ").strip()
    if not query:
      continue
    print("ClimateAI: Thinking...\n")
    print(generate_answer(query))
    print()
  except KeyboardInterrupt:
    print("\nExiting.")
    break