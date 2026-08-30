import os
import chromadb
from config import CHROMA_DB_DIR, LEXICON_FILE_PATH, RAG_NUM_RESULTS

class LexiconRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection(name="wh40k_lexicon")
        
        if self.collection.count() == 0:
            self.load_lexicon()

    def load_lexicon(self):
        if not os.path.exists(LEXICON_FILE_PATH):
            print(f"[WARNING] Lexicon file '{LEXICON_FILE_PATH}' not found.")
            return

        print(f"[RAG] Indexing '{LEXICON_FILE_PATH}' into local vector database...")
        with open(LEXICON_FILE_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        documents = [line for line in lines]
        ids = [f"doc_{idx}" for idx in range(len(lines))]

        if documents:
            self.collection.add(documents=documents, ids=ids)
            print(f"[RAG] Successfully indexed {len(documents)} entries.")

    def search(self, query, n_results=RAG_NUM_RESULTS):
        if self.collection.count() == 0:
            return ""

        results = self.collection.query(query_texts=[query], n_results=n_results)
        retrieved_docs = results.get("documents", [[]])[0]
        return "\n".join(retrieved_docs)