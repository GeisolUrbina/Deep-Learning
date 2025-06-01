# --------------------------
# Streamlit-app för Bibel-Chatbot
# --------------------------

import os
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import zipfile
import gdown

# --------------------------
# KONFIGURATION & SÄKERHET 
# --------------------------

try:
    # Försök hämta från Streamlit Secrets först
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback till .env-fil för lokal utveckling
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ API-nyckel saknas. Konfigurera den i Secrets eller .env-fil.")
        st.stop()

openai.api_key = api_key

# --------------------------
# IMPORTERA BIBLIOTEK OCH MODULER
# --------------------------

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
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
# OPTIMERAD FUNKTION FÖR ATT LADDA FAISS-INDEX
# --------------------------

@st.cache_resource
def load_retriever():
    """
    Ladda FAISS-index från Google Drive
    med bättre felhantering och länkhantering.
    """
    # Konfiguration
    GOOGLE_DRIVE_ID = "1fDt5WhZPV_C-u5XM5tzaI5TSaxnLuL31"
    INDEX_PATH = "data/faiss_index"
    ZIP_PATH = "temp_data.zip"  # Använder temporär filnamn
    
    # Skapa mappstruktur om den inte finns
    os.makedirs(INDEX_PATH, exist_ok=True)
    
    # Kontrollera om index redan finns
    if not all(os.path.exists(f"{INDEX_PATH}/{f}") for f in ["index.faiss", "index.pkl"]):
        with st.spinner("🔄 Hämtar kunskapsbas från Google Drive..."):
            try:
                # Använd korrekt nedladdnings-URL format
                download_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}&export=download"
                
                # Ladda ner med progress indicator
                gdown.download(
                    download_url,
                    ZIP_PATH,
                    quiet=False
                )
                
                # Verifiera och extrahera ZIP
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    # Extrahera specifikt till INDEX_PATH
                    for file in zip_ref.namelist():
                        if file.startswith("faiss_index/"):
                            zip_ref.extract(file, "data")
                
                # Rensa upp ZIP-fil efter extrahering
                os.remove(ZIP_PATH)
                
                # Final verifiering
                if not all(os.path.exists(f"{INDEX_PATH}/{f}") for f in ["index.faiss", "index.pkl"]):
                    raise FileNotFoundError("Nödvändiga indexfiler saknas efter extrahering")
                    
            except zipfile.BadZipFile:
                st.error("""
                ❌ Ogiltig ZIP-fil. Möjliga orsaker:
                1. Felaktig Google Drive-länk
                2. Filen är korrupt
                3. Behörighetsproblem
                """)
                st.stop()
            except Exception as e:
                st.error(f"⛔ Kritisk fel: {str(e)}")
                if os.path.exists(ZIP_PATH):
                    os.remove(ZIP_PATH)
                st.stop()
    
    # Ladda FAISS-index
    try:
        embeddings = OpenAIEmbeddings()
        store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("✅ Kunskapsbas laddad!")
        return store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"🔴 Fel vid laddning av FAISS-index: {str(e)}")
        st.stop()
        
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
        {"role": "assistant", "content": "Hej! Jag är din Bibel-Chatbot. Fråga gärna om något bibelställe eller tema, så hjälper jag dig!"}
    ]

# --------------------------
# RENDERA MESSAGES SOM CHATTBUBBLOR
# --------------------------

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --------------------------
# INPUTFÄLT FÖR ANVÄNDARENS FRÅGA
# --------------------------

if user_input := st.chat_input("Skriv din fråga här..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        answer = qa_chain.run(user_input)
        if not answer.strip():
            answer = "Jag förstår inte riktigt. Kan du formulera frågan på ett annat sätt?"
    except Exception as e:
        answer = "❌ Ett fel uppstod vid generering av svaret. Försök igen senare."
        st.error(f"Detaljerat fel: {e}")

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
