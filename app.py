# --------------------------
# Streamlit-app för Bibel-Chatbot
# --------------------------

import os
import zipfile
from dotenv import load_dotenv
import streamlit as st
import openai
import gdown

# --------------------------
# KONFIGURATION & SÄKERHET
# --------------------------

# Försök ladda en eventuell .env-fil (fungerar lokalt, men gör inget om fil saknas)
load_dotenv()

# Hämta API-nyckeln från miljön (eller från Streamlit Secrets på Cloud)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY saknas i miljön. Lägg in den i en lokal .env eller i Streamlit Secrets.")
    st.stop()

# Sätt API-nyckeln för openai-paketet
openai.api_key = api_key

# --------------------------
# IMPORTERA BIBLIOTEK OCH MODULER
# --------------------------

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --------------------------
# SIDKONFIGURATION OCH DESIGN
# --------------------------

st.set_page_config(
    page_title="📖 Bibeln RAG-Chatbot",
    layout="wide"
)

# --------------------------
# SKAPA EGNA PROMPT-TEMPLATE
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
# FUNKTION FÖR ATT LADDA FAISS-INDEX FRÅN GOOGLE DRIVE
# --------------------------

@st.cache_resource
def load_retriever():
    """
    Ladda FAISS-index från Google Drive om det inte finns lokalt,
    och skapa en retriever-objekt.
    """
    # Kontrollera om indexet redan finns
    index_path = "data/faiss_index"
    if not os.path.exists(index_path):
        with st.spinner("Laddar FAISS-index från Google Drive..."):
            # Skapa mapp om den inte finns
            os.makedirs("data", exist_ok=True)
            
            # Ladda ner komprimerad fil från Google Drive
            zip_path = "faiss_index.zip"
            gdown.download(
                "https://drive.google.com/uc?id=DIN_GOOGLE_DRIVE_ID_HÄR",
                zip_path,
                quiet=False
            )
            
            # Packa upp filen
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data/")
            
            # Ta bort zip-filen
            os.remove(zip_path)
    
    # Ladda indexet
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_kwargs={"k": 3})

# --------------------------
# LÄS IN FAISS-INDEXET
# --------------------------

with st.spinner("Laddar kunskapsbas..."):
    retriever = load_retriever()

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
            "content": "Hej! Jag är din Bibel-Chatbot. Fråga gärna om något bibelställe eller tema, så hjälper jag dig!"
        }
    ]

# --------------------------
# RENDERA MESSAGES SOM CHATTBUBBLOR
# --------------------------

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --------------------------
# INPUTFÄLT FÖR ANVÄNDARENS FRÅGA
# --------------------------

if user_input := st.chat_input("Skriv din fråga här..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Anropa QA-kedjan för att få svaret
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
