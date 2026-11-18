import os
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_NAME = "healthcare_faiss"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200


def load_environment() -> None:
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("New_OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or New_OPENAI_API_KEY in .env or environment variables."
        )
    os.environ["OPENAI_API_KEY"] = key


def load_documents(documents_path: Path = DOCS_DIR):
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_path}")

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }
    documents = []
    for file_path in sorted(documents_path.iterdir()):
        if file_path.suffix.lower() not in loaders:
            continue
        loader_cls = loaders[file_path.suffix.lower()]
        loader = loader_cls(str(file_path))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        documents.extend(loaded)

    if not documents:
        raise FileNotFoundError(
            f"No supported documents found in {documents_path}. Add .pdf or .txt files."
        )
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    split_docs = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_index"] = chunk_index
            if "page" in doc.metadata and "page" not in chunk.metadata:
                chunk.metadata["page"] = doc.metadata["page"]
            if "page_number" in doc.metadata and "page_number" not in chunk.metadata:
                chunk.metadata["page_number"] = doc.metadata["page_number"]
            split_docs.append(chunk)
    return split_docs


def build_vector_store():
    load_environment()
    documents = load_documents()
    chunks = split_documents(documents)
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        vector_store.save_local(str(VECTOR_STORE_DIR), index_name=INDEX_NAME)
        print(f"Ingested {len(documents)} documents into {len(chunks)} chunks.")
        print(f"Saved FAISS vector store to {VECTOR_STORE_DIR}")
        return vector_store
    except Exception as exc:
        # On Windows long-paths or special filesystem issues FAISS may fail to write.
        # Fall back to a shorter path in the user's home directory and record the location.
        from pathlib import Path
        fallback_dir = Path.home() / ".rag_vector_store"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        try:
            vector_store.save_local(str(fallback_dir), index_name=INDEX_NAME)
            note_path = VECTOR_STORE_DIR / "SAVE_FALLBACK.txt"
            note_path.write_text(f"FAISS save to project folder failed: {exc}\nSaved to fallback: {fallback_dir}\n")
            print(f"Saved FAISS vector store to fallback location: {fallback_dir}")
            return vector_store
        except Exception as exc2:
            print("Failed to save FAISS index to both project and fallback locations.")
            raise exc2


def load_vector_store():
    load_environment()
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python ingest.py --reindex` first."
        )

    # If a fallback save was used, a SAVE_FALLBACK.txt file will point to the real location.
    fallback_note = VECTOR_STORE_DIR / "SAVE_FALLBACK.txt"
    if fallback_note.exists():
        # Read using latin-1 to avoid unicode decode errors on unexpected bytes
        with open(fallback_note, "r", encoding="latin-1", errors="ignore") as fh:
            content = fh.read()
        # Attempt to extract path after 'Saved to fallback:'
        for line in content.splitlines():
            if "Saved to fallback:" in line:
                fallback_path = line.split("Saved to fallback:", 1)[1].strip()
                if fallback_path:
                    return FAISS.load_local(str(fallback_path), OpenAIEmbeddings(model=EMBEDDING_MODEL), index_name=INDEX_NAME, allow_dangerous_deserialization=True)

    return FAISS.load_local(str(VECTOR_STORE_DIR), OpenAIEmbeddings(model=EMBEDDING_MODEL), index_name=INDEX_NAME, allow_dangerous_deserialization=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest healthcare documents and build a FAISS vector store.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the vector store from docs.")
    args = parser.parse_args()

    if args.reindex:
        build_vector_store()
    else:
        print("Use --reindex to create or refresh the vector store from the docs folder.")


if __name__ == "__main__":
    main()
