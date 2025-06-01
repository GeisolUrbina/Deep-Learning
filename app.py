# --------------------------
# Streamlit-app för Bibel-Chatbot
# --------------------------

import os
import json
import requests
import time
from dotenv import load_dotenv
import streamlit as st
import openai

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --------------------------
# KONFIGURATION & SÄKERHET
# --------------------------

# Försök ladda en eventuell .env-fil (fungerar lokalt, men gör inget om fil saknas)
load_dotenv()

# Hämta API-nyckeln från miljön (eller från Streamlit Secrets på Cloud)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY saknas i miljön. Lägg in den i Streamlit Secrets eller i en lokal .env.")
    st.stop()

# Sätt API-nyckeln för openai-paketet
openai.api_key = api_key

# --------------------------
# SIDKONFIGURATION OCH DESIGN
# --------------------------

st.set_page_config(
    page_title="📖 Bibeln RAG-Chatbot",
    layout="wide"
)

# --------------------------
# FUNKTION FÖR ATT BYGGA FAISS-INDEX OM DET SAKNAS
# --------------------------

def build_faiss_index():
    """
    Hämtar alla bibelböcker via Bible API, chunkar text,
    skapar embeddings och sparar FAISS-indexet i data/faiss_index/.
    Körs bara om indexmappen saknas.
    """
    # Läs in alla böcker/kapitel från books.json
    with open("books.json", "r", encoding="utf-8") as f:
        BOOKS = json.load(f)

    documents = []
    for book, chapters in BOOKS.items():
        for chap in range(1, chapters + 1):
            url = f"https://bible-api.com/{book}%20{chap}"
            while True:
                r = requests.get(url)
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 5))
                    st.write(f"Rate limit på {book} kapitel {chap}, väntar {retry_after}s…")
                    time.sleep(retry_after)
                    continue
                r.raise_for_status()
                break

            text = r.json().get("text", "")
            documents.append(Document(page_content=text,
                                      metadata={"book": book, "chapter": chap}))
            time.sleep(0.2)  # Paus mellan anrop

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    st.write(f"Totalt chunkar: {len(chunks)}")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    index = FAISS.from_documents(chunks, embeddings)

    os.makedirs("data/faiss_index", exist_ok=True)
    index.save_local("data/faiss_index")
    st.write("🔥 FAISS-index byggt och sparat i data/faiss_index/")

# --------------------------
# BYGG INDEX OM DEN SAKNAS
# --------------------------

if not os.path.isdir("data/faiss_index"):
    st.sidebar.info("📥 Bygger om FAISS-index – detta kan ta ett par minuter.")
    build_faiss_index()
else:
    st.sidebar.success("✅ FAISS-index finns, används som det är.")

# --------------------------
# FUNKTION FÖR ATT LADDAR FAISS-INDEX
# --------------------------

@st.cache_resource
def load_retriever():
    """
    Ladda FAISS-index och skapa en retriever-objekt.
    Returnerar en retriever som kan användas för likhetssökning.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    store = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_kwargs={"k": 3})

# ──────────────────────────────────────────────────────────────────────────
# LÄS IN FAISS-INDEXET OCH BYGG RETRIEVER
# ──────────────────────────────────────────────────────────────────────────

with st.spinner("Laddar FAISS-retriever…"):
    retriever = load_retriever()

# --------------------------
# SKAPA PROMPT-TEMPLATE
# --------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Du är en vänlig och hjälpsam bibelguide som svarar på frågor genom att använda givna bibelavsnitt.
Om du inte hittar relevant information ska du be användaren om att skriva om frågan.

Kontekst:
{context}

Fråga:
{question}

Svar:
"""
)

# --------------------------
# INITIERA LLM OCH QA-KEDJA
# --------------------------

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# --------------------------
# KONVERSATIONS-SESSIONSTATE
# --------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hej! Jag är din Bibel-Chatbot. Fråga gärna om något bibelställe eller tema!"
        }
    ]

# --------------------------
# RENDERA MESSAGES SOM CHATTBUBBLOR
# --------------------------

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --------------------------
# INPUTFÄLT FÖR ANVÄNDARENS FRÅGA
# --------------------------

if user_input := st.chat_input("Skriv din fråga här…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        answer = qa_chain.run(user_input)
    except Exception as e:
        answer = "❌ Ett fel uppstod vid generering av svaret. Försök igen senare."
        st.error(f"Detaljerat fel: {e}")

    if not answer.strip():
        answer = "Jag förstår inte riktigt. Kan du formulera frågan på ett annat sätt?"
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

# --------------------------
# FOOTER
# --------------------------

st.markdown("---")
st.caption("""
📖 *Svenska Bibel-Chatbot v1.0* | 
Datakälla: Svenska Bibelsällskapet | 
Byggd med Python & LangChain
""")
