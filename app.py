@st.cache_resource
def load_retriever():
    """
    Ladda FAISS-index från Google Drive
    med bättre felhantering och extrahering.
    """
    import os
    import zipfile
    import gdown
    from langchain.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores.faiss import FAISS

    # Konfiguration
    GOOGLE_DRIVE_ID = "1fDt5WhZPV_C-u5XM5tzaI5TSaxnLuL31"
    INDEX_PATH = "data/faiss_index"
    ZIP_PATH = "temp_data.zip"

    # Skapa mappstruktur om den inte finns
    os.makedirs("data", exist_ok=True)

    # Kontrollera om index redan finns
    if not all(os.path.exists(f"{INDEX_PATH}/{f}") for f in ["index.faiss", "index.pkl"]):
        with st.spinner("🔄 Hämtar kunskapsbas från Google Drive..."):
            try:
                # Ladda ner zip från Google Drive
                download_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}&export=download"
                gdown.download(download_url, ZIP_PATH, quiet=False)

                # Extrahera hela zip-filen till ./data
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall("data")

                # Rensa upp zip-filen
                os.remove(ZIP_PATH)

                # Debug: visa extraherade filer (valfritt)
                for root, dirs, files in os.walk("data"):
                    for file in files:
                        st.write("📁 Extraherad fil:", os.path.join(root, file))

                # Verifiera att nödvändiga indexfiler finns
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

    # Ladda FAISS-index med embeddings
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
