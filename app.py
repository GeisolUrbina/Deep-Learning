# --------------------------
# Streamlit-app för Bibel-Chatbot
# --------------------------

import os
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


# --------------------------
# IMPORTERA BIBLIOTEK OCH MODULER
# --------------------------

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --------------------------
# SIDKONFIGURATION OCH DESIGN
# --------------------------

st.set_page_config(
    page_title="📖 Bibeln RAG-Chatbot", 
    layout="wide"                       
)


# --------------------------
# TITEL OCH INTRODUKTION
# --------------------------

st.title("📖 Bibel-Chatbot  -  Fråga om Bibeln")


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
    Ladda FAISS-index från Google Drive, extrahera allt under ./data
    och hitta automagiskt den mapp som innehåller index.faiss + index.pkl.
    """
    import os
    import zipfile
    import gdown

    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    # Google Drive-ID för ZIP:en som innehåller faiss_index/
    GOOGLE_DRIVE_ID = st.secrets["GOOGLE_DRIVE_ID"]
    ZIP_PATH = "temp_data.zip"
    BASE_DIR = "data"  # där vi extraherar allt

    # Se till att data-mappen finns
    os.makedirs(BASE_DIR, exist_ok=True)

    # Kontrollera om vi redan har index-filerna
    def hitta_index_mapp():
        """
        Leta igenom BASE_DIR och returnera första mapp som har
        både index.faiss och index.pkl.
        Returnerar sökvägen (str) om hittad, annars None.
        """
        for root, dirs, files in os.walk(BASE_DIR):
            if "index.faiss" in files and "index.pkl" in files:
                return root
        return None

    befintlig_mapp = hitta_index_mapp()
    if befintlig_mapp:
        index_path = befintlig_mapp
    else:
        # Ladda och extrahera ZIP’en
        with st.spinner("🔄 Hämtar kunskapsbas från Google Drive..."):
            try:
                download_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}&export=download"
                gdown.download(download_url, ZIP_PATH, quiet=False)

                with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                    zip_ref.extractall(BASE_DIR)

                os.remove(ZIP_PATH)

                index_path = hitta_index_mapp()

                if not index_path:
                    raise FileNotFoundError("Nödvändiga indexfiler saknas efter extrahering.")

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

    # Ladda FAISS-index med embeddings
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key   
        )

        store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        st.success("Välkommen!")
        return store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"🔴 Fel vid laddning av FAISS-index: {str(e)}")
        st.stop()
        

# --------------------------
# FUNKTION FÖR ATT SPARA FEEDBACK TILL CSV
# --------------------------

import csv
from datetime import datetime

def spara_feedback(fraga, svar, feedback):
    filnamn = "feedback_logg.csv"
    tidpunkt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rad = [tidpunkt, fraga, svar, feedback]

    # Skapa fil med rubriker om den inte redan finns
    if not os.path.exists(filnamn):
        with open(filnamn, mode="w", newline="", encoding="utf-8") as fil:
            writer = csv.writer(fil)
            writer.writerow(["Tidpunkt", "Fråga", "Svar", "Feedback"])
            writer.writerow(rad)
    else:
        with open(filnamn, mode="a", newline="", encoding="utf-8") as fil:
            writer = csv.writer(fil)
            writer.writerow(rad)


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
    temperature=0,
    api_key=api_key     
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

        # Endast om svaret är lyckat, spara för feedback
        st.session_state["senaste_fraga"] = user_input
        st.session_state["senaste_svar"] = answer

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



