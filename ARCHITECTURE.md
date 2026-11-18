# Healthcare RAG Assistant Architecture

## Overview

This project is a Retrieval-Augmented Generation (RAG) application for healthcare domain QA.
It uses a set of healthcare PDF documents as the only source of truth and answers user questions by retrieving relevant passages, then generating grounded responses via OpenAI.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌────────────────────┐                  ┌──────────────────┐   │
│  │   Streamlit UI     │                  │   CLI (app.py)   │   │
│  │  - Chat Interface  │                  │  - Single Query  │   │
│  │  - Retrieval View  │                  │  - Reindex Mode  │   │
│  │  - Feedback        │                  │  - Chat Loop     │   │
│  └────────────────────┘                  └──────────────────┘   │
└─────────────────────┬──────────────────────────────┬────────────┘
                      │                              │
                      └──────────────┬───────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   HealthcareRAGAssistant         │
                    │   (chatbot.py)                   │
                    │ - Conversation History           │
                    │ - Answer Generation              │
                    │ - Source Citation                │
                    └────────────┬──────────┬──────────┘
                                 │          │
                    ┌────────────▼──┐    ┌─▼────────────────┐
                    │  Retriever    │    │  LLM (GPT-3.5)   │
                    │  (FAISS)      │    │  (OpenAI)        │
                    │  - Top-K      │    │  - Generation    │
                    │  - Scoring    │    │  - Grounding     │
                    └────────┬──────┘    └──────────────────┘
                             │
                    ┌────────▼──────────────────┐
                    │  FAISS Vector Store       │
                    │  ~/.rag_vector_store (*)  │
                    │  - 768-dim embeddings     │
                    │  - Semantic indexing      │
                    └────────▲──────────────────┘
                             │
                    ┌────────┴──────────────────┐
                    │  Document Ingestion       │
                    │  (ingest.py)              │
                    │ - Load PDFs/TXT           │
                    │ - Chunking (900 tokens)   │
                    │ - Embedding (text-embed-3)│
                    └────────▲──────────────────┘
                             │
                    ┌────────┴──────────────────┐
                    │  docs/ (PDF Documents)    │
                    │  - Visiting Hours FAQ     │
                    │  - Insurance Coverage FAQ │
                    │  - Hypertension Guide     │
                    │  - Asthma Guide           │
                    │  - Medications Guide      │
                    └───────────────────────────┘

(*) Falls back to ~/.rag_vector_store on Windows path issues
```

## Quick Component Overview

| Component | File | Role |
|-----------|------|------|
| **Ingestion** | `ingest.py` | Load docs → split → embed → store in FAISS |
| **RAG Chain** | `chatbot.py` | Retrieve context + chat history → generate answer |
| **Web UI** | `streamlit_app.py` | Browser interface for conversation & diagnostics |
| **CLI** | `app.py` | Command-line access to reindex/query |
| **Utilities** | `setup_project.py` | Demo PDFs & environment setup |

## Detailed Component Description

1. **Document Ingestion**
   - `ingest.py`
   - Loads PDF and text documents from `docs/`.
   - Splits documents into chunks using `RecursiveCharacterTextSplitter`.
   - Creates embeddings with `OpenAIEmbeddings`.
   - Stores the vector index in a local FAISS store.
   - Includes fallback save logic to `~/.rag_vector_store` for Windows path compatibility.

2. **Conversational RAG Chain**
   - `chatbot.py`
   - Loads the vector store and builds a retriever.
   - Uses `ChatOpenAI` and `ConversationalRetrievalChain`.
   - Supports conversation history via `ConversationBufferMemory`.
   - Enforces a refusal response when the answer is not grounded in retrieved text.
   - Returns answer text, source citations, and metadata.

3. **Streamlit UI**
   - `streamlit_app.py`
   - Provides a browser-based interface for query input, conversation management, retrieval diagnostics, and feedback.
   - Allows users to rebuild the vector store, ask questions, reset conversation, and inspect retrieved passages.
   - Shows source citations separately to avoid duplicate source rendering.

4. **CLI Wrapper**
   - `app.py`
   - Provides a quick command-line way to reindex, open a chat loop, or answer a single question.

5. **Project Utilities**
   - `setup_project.py`
   - Generates sample PDF documents and initializes a local `.env` file during demo setup.

## Data Flow

1. **Document ingestion**
   - Source PDFs are loaded from `docs/`.
   - Documents are split into overlapping semantic chunks.
   - Chunks are embedded and stored in FAISS.

2. **Query handling**
   - User asks a question via Streamlit or CLI.
   - The question is sent to the retriever.
   - Retriever returns top-K relevant chunks from FAISS.
   - The conversational chain combines retrieved context with chat history.
   - The LLM generates an answer using only the retrieved context.

3. **Response and grounding**
   - The answer is returned with cited source documents.
   - If no sufficient evidence exists, the chain returns:
     - `I don’t have enough information in the provided documents.`

## Core Files

- `ingest.py` — ingestion, splitting, embedding, vector store build/load
- `chatbot.py` — conversational chain, retrieval, feedback, source extraction
- `streamlit_app.py` — browser UI, conversation state, retrieval diagnostics
- `app.py` — CLI interface for reindexing, chat, and single-question queries
- `requirements.txt` — runtime dependencies
- `requirements-dev.txt` — optional developer tooling
- `README.md` — usage and setup guidance
- `ARCHITECTURE.md` — this architecture document

## Dependencies

Key runtime dependencies:
- `openai`
- `langchain`
- `langchain-community`
- `langchain-openai`
- `streamlit`
- `python-dotenv`
- `faiss-cpu`
- `pypdf`

## Environment

- Uses `.env` for `OPENAI_API_KEY` and `New_OPENAI_API_KEY`.
- `.env` is excluded from version control via `.gitignore`.
- The application can read either env variable.

## Persistence

- The FAISS index is stored locally in `vector_store/` when possible.
- If local save fails, the app falls back to `C:\Users\<user>\.rag_vector_store`.
- This fallback ensures Windows path compatibility and prevents broken ingestion.

## Notes on Hallucination Control

- The prompt template explicitly instructs the model to use only the retrieved context.
- The app includes an explicit refusal fallback when source evidence is insufficient.
- Streamlit shows source documents and retrieval metrics to make grounding visible.

## Recommended Execution

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create `.env` with your OpenAI key.
3. Rebuild the vector store:
   ```bash
   python ingest.py --reindex
   ```
4. Run the UI:
   ```bash
   streamlit run streamlit_app.py
   ```

## Future Improvements

- Add a dedicated retriever-only validation mode.
- Add automated hallucination regression tests.
- Replace the current LangChain imports with fully supported API versions if dependencies change.
