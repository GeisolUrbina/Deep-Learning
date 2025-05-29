#  🤖📖 Bibel-Chatbot (RAG + Streamlit)

En interaktiv chatbot som använder Retrieval-Augmented Generation (RAG) för att svara på frågor om Bibeln.  
Bygger en FAISS-vektorindex av Bibeltexten via Bible API och använder OpenAI-modeller för semantiskt QA. Skalad med Streamlit för enkel web-UI.

---

## 🚀 Funktioner

- 🔍 **Naturligt språk**  
  Sök i hela Bibeln med fri text.
  
- ⚙️ **RAG-pipeline**  
  API-hämtning → chunking → embeddings → FAISS-index.
  
- 🌐 **Streamlit-app**  
  Enkelt och användarvänligt gränssnitt direkt i webbläsaren.
  
---
## 🚀 Kom igång!

### 1. Klona repot
```bash
git clone https://github.com/DITT-ANVÄNDARNAMN/bibel-chatbot.git
cd bibel-chatbot
```
### 2. Skapa virtuell miljö & installera beroenden
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Lägg in din OpenAI-nyckel
1. Kopiera ```.env.example``` till ```.env```
```bash
cp .env.example .env
```
2. Öppna ```.env``` och fyll i din egen nyckel:
```bash
OPENAI_API_KEY=din_openai_api_nyckel
```

### 4. Bygg FAISS-index
```bash
python ingest_api.py
```
### 5. Starta Streamlit-appen
```bash
streamlit run app.py
```

---
### ✨ Rekommendationer för optimal användning

- **Formulera tydliga och precisa frågor**: Ju mer konkret din fråga är, desto mer relevant och exakt blir svaret.  
- **Använd vedertagna referenser**: Använd t.ex. “John 3:16” snarare än “Johannes 3:16” för säkrare uppslag.  
- **Utforska olika teman**: Be om historisk bakgrund, parallelltexter, sammanfattningar eller teologiska reflektioner för djupare insikt.  


---
© 2025 Geisol Urbina | EC Utbildning. Alla rättigheter förbehållna.


