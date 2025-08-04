import os, json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from tqdm import tqdm

# CONFIG
ARTICLES_DIR = "./data/processed"
CHROMA_DIR = "./src/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_articles():
  docs = []
  for file in os.listdir(ARTICLES_DIR):
    with open (os.path.join(ARTICLES_DIR, file), "r", encoding="utf-8") as f:
      articles = json.load(f)
    for article in articles:
      content = article.get("content", "")
      if content.strip():
        docs.append(Document(page_content=content))
  return docs

print("😒Loading articles...")
docs = load_articles()

print(f"👌Loaded {len(docs)} articles. Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print("😍Embedding and storing in ChromaDB...")
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma.from_documents(chunks, embedding_fn, persist_directory=CHROMA_DIR)
vectordb.persist()

print(f"👍CHROMADB built with {len(chunks)} chunks at {CHROMA_DIR}")