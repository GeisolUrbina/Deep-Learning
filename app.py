# --------------------------
# Streamlit-app för Bibel-Chatbot
# --------------------------

import os
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv

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

# LangChain-klasser för embeddings, vektorlagring, chat-modell, prompt och kedja.
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --------------------------
# SIDKONFIGURATION OCH DESIGN
# --------------------------

# Sätt sidinställningar för Streamlit-appen
st.set_page_config(
    page_title="📖 Bibeln RAG-Chatbot", 
    layout="wide"                       
)

# --------------------------
# SKAPA EGNA PROMPT-TEMPLATE
# --------------------------

# Använd en PromptTemplate för att definiera hur frågan och kontexten ska formateras.
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
# FUNKTION FÖR ATT LADDAR FAISS-INDEX
# --------------------------

@st.cache_resource
def load_retriever(index_path: str):
    """
    Ladda FAISS-index och skapa en retriever-objekt.

    Parametrar:
    - index_path: Sökväg till mappen där FAISS-index har sparats.

    Returnerar:
    - En retriever som kan användas för likhetssökning.
    """
    # Skapa en embeddings-instans (OpenAI-embeddings hämtar API-nyckel från miljö)
    embeddings = OpenAIEmbeddings()

    # Ladda indexet från disk
    store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Returnera retriever med fast antal källor (k=3)
    return store.as_retriever(search_kwargs={"k": 3})

# --------------------------
# LÄS IN FAISS-INDEXET
# --------------------------

# Visa en spinner medan kunskapsbasen laddas (FAISS-indexet kan ta en stund)
with st.spinner("Laddar kunskapsbas..."):
    retriever = load_retriever("data/faiss_index")

# --------------------------
# INITIERA LLM OCH QA-KEDJA
# --------------------------

# ChatOpenAI: Wrapper för OpenAI:s chat-modell (gpt-3.5-turbo)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  
    temperature=0                
)

# Skapa RetrievalQA-kedjan med “stuff”-metoden och egen prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",               # Enkel kedja som samlar in hela kontexten
    retriever=retriever,
    return_source_documents=False,     # False, för att inte vill visa källor i svaret
    chain_type_kwargs={"prompt": prompt}  
)

# --------------------------
# KONVERSATIONS-SESSIONSTATE
# --------------------------

# Använd Streamlit session_state för att spara tidigare meddelanden 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej! Jag är din Bibel-Chatbot. Fråga gärna om något bibelställe eller tema, så hjälper jag dig!"}
    ]

# --------------------------
# RENDERA MESSAGES SOM CHATTBUBBLOR
# --------------------------

# Loopa igenom alla lagrade meddelanden och visa dem med st.chat_message
for msg in st.session_state.messages:
    if msg["role"] == "user":
        
        st.chat_message("user").write(msg["content"])
    else:
        # Visa chatbotens svar som en assistent-bubble
        st.chat_message("assistant").write(msg["content"])

# --------------------------
# INPUTFÄLT FÖR ANVÄNDARENS FRÅGA
# --------------------------

# st.chat_input visar ett textfält med chattliknande stil
if user_input := st.chat_input("Skriv din fråga här..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Anropa QA-kedjan för att få svaret
    answer = qa_chain.run(user_input)

    # Hantera tomma eller irrelevanta svar
    if not answer.strip():
        # Ge en standardfeedback om inget svar hittas
        answer = "Jag förstår inte riktigt. Kan du formulera frågan på ett annat sätt?"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    else:
        # Spara och visa det genererade svaret
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

# --------------------------
# FOOTER
# --------------------------

# Visa en enkel footer med info om appens verktyg

st.markdown("---")
st.caption("""
📖 *Svenska Bibel-Chatbot v1.0* | 
Datakälla: Svenska Bibelsällskapet | 
Byggd med Python & LangChain
""")
