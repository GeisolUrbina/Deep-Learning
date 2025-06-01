# --------------------------
# Streamlit-app för Bibel-Chatbot
# --------------------------

import os
from dotenv import load_dotenv
import streamlit as st
import openai

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
    retriever = load_retriever("data/faiss_index")

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
