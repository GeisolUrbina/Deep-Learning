import os
import json
import time
import requests
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Läs in din OpenAI-nyckel
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Ladda in böcker och kapitel
with open('books.json', 'r', encoding='utf-8') as f:
    BOOKS = json.load(f)

docs = []
for book, chapters in BOOKS.items():
    for chap in range(1, chapters + 1):
        url = f'https://bible-api.com/{book}%20{chap}'
        
        # Hantera rate-limit (429)
        while True:
            resp = requests.get(url)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get('Retry-After', 5))
                print(f"429: rate limit på {book} kapitel {chap}, väntar {retry_after}s…")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            break

        # Extrahera verser
        data = resp.json()
        text = ' '.join(v.get('text', '') for v in data.get('verses', []))
        docs.append(Document(page_content=text, metadata={'book': book, 'chapter': chap}))
        print(f'Hämtade {book} kapitel {chap}')

        # Kort paus mellan anrop för att minskar risken att träffa rate-limit.
        time.sleep(0.2)

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f'Totalt chunkar: {len(chunks)}')

# Embeddings + FAISS-index
emb = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = FAISS.from_documents(chunks, emb)

# Spara index
index_dir = 'data/faiss_index'
os.makedirs(index_dir, exist_ok=True)
vector_store.save_local(index_dir)
print(f'Index sparat i {index_dir}')
