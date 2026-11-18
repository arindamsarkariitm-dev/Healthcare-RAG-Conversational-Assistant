# Healthcare RAG Conversational Assistant

This repository implements a domain-specific healthcare assistant using a Retrieval-Augmented Generation pipeline.
It retrieves relevant passages from PDF patient education guides and hospital FAQs, then answers using only that retrieved content.

## Architecture

- **Document Ingestion**: PDF documents are loaded from `docs/`, split into semantic chunks, embedded via OpenAI, and stored in a local FAISS vector store.
- **Retrieval Agent**: Queries are matched against the vector store to retrieve top-k relevant chunks.
- **Generation Agent**: A LangChain conversational chain uses retrieved context and chat history to answer follow-up questions.
- **Safety Logic**: If retrieval confidence is low or the answer cannot be grounded in the documents, the assistant returns:
  - `I don’t have enough information in the provided documents.`
- **Feedback**: User feedback is logged in `feedback.log`.

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The requirements file now includes the newer LangChain adapter packages:
- `langchain-community`
- `langchain-openai`

These packages are required because the code uses the modern `langchain_community` and `langchain_openai` imports instead of the older `langchain` package alone.

Optional developer tools can be installed with:

```bash
pip install -r requirements-dev.txt
```

3. Create a local `.env` file and add your OpenAI key. For example, open a new `.env` file and add:

```text
OPENAI_API_KEY=your_key_here
New_OPENAI_API_KEY=your_key_here
```

Then save `.env` in the project root.

> Note: `.env` is excluded from version control by `.gitignore`, so it is safe to keep secrets there.

## Documents

The `docs/` directory contains the required PDF source files. These documents are used directly for retrieval and citation.

## Usage

Build or refresh the vector store:

```bash
python ingest.py --reindex
```

Start the web UI:

```bash
streamlit run streamlit_app.py
```

Start the CLI chatbot:

```bash
python app.py --chat
```

Ask one question directly from the CLI:

```bash
python app.py --question "What are the visiting hours for family members?"
```

## Notes

- The assistant is not fine-tuned; it uses retrieval to ground responses.
- Answers include source citations from the provided PDFs.
- Follow-up questions are supported through conversation history.

## Tests

Run the optional developer tests with:

```bash
pytest tests
```

## Sample Log

Run the assistant and inspect `sample_conversation.txt` for a sample interaction.
