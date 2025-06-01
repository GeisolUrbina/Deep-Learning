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

# Ladda .env-fil och kontrollera API-nyckeln
env_path = find_dotenv()
if not env_path:
    raise FileNotFoundError(
        "⚠️ .env-fil inte hittad. Se till att den ligger i projektets rotkatalog."
    )
load_dotenv(env_path)

# Hämta API-nyckel
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("🔑 API-nyckel saknas i .env-filen. Kontrollera att OPENAI_API_KEY är definierad.")
# Sätt API-nyckeln för `openai`-paketet
oai = openai
oai.api_key = api_key

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
# FUNKTION FÖR ATT LADDA FAISS-INDEX
# --------------------------

@st.cache_resource
def load_retriever():
    """
    Ladda FAISS-index från Google Drive om det inte finns lokalt,
    och skapa en retriever-objekt.
    """
    index_path = "data/faiss_index"
    zip_path = "data.zip"
    
    if not os.path.exists(index_path):
        with st.spinner("Förbereder kunskapsbas..."):
            # Skapa mapp om den inte finns
            os.makedirs("data", exist_ok=True)
            
            # Ladda ner från Google Drive om zip-fil saknas
            if not os.path.exists(zip_path):
                try:
                    gdown.download(
                        "https://drive.google.com/file/d/1bdspw4vRQ6Ui0oaTE91B5WyVEz0vD8cA/view?usp=drive_link",
                        zip_path,
                        quiet=False
                    )
                except Exception as e:
                    st.error(f"Kunde inte ladda ner fil: {str(e)}")
                    st.stop()
            
            # Verifiera och packa upp zip-filen
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("data/")
            except zipfile.BadZipFile:
                st.error("Filen är inte en giltig ZIP. Kontrollera Google Drive-länken.")
                st.stop()
            except Exception as e:
                st.error(f"Fel vid uppackning: {str(e)}")
                st.stop()
    
    # Ladda FAISS-index
    try:
        embeddings = OpenAIEmbeddings()
        store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Fel vid laddning av FAISS-index: {str(e)}")
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