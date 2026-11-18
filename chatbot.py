import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from ingest import load_environment, load_vector_store

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_LOG = BASE_DIR / "feedback.log"

PROMPT_TEMPLATE = """
You are a healthcare domain assistant for patient education and hospital FAQs.
Use only the provided retrieved documents in your answer.
If the answer cannot be found in the provided context, respond exactly:
I don’t have enough information in the provided documents.

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)


class HealthcareRAGAssistant:
    def __init__(self, top_k: int = 5):
        load_environment()
        self.vector_store = load_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
        )

    def chat(self, question: str) -> Dict[str, object]:
        if not question.strip():
            return {"answer": "Please enter a valid question."}

        result = self.chain({"question": question})
        answer_text = result.get("answer", "").strip()
        source_documents = result.get("source_documents", [])
        citations = self._extract_sources(source_documents)

        return {
            "answer": answer_text,
            "sources": sorted(citations),
            "source_documents": source_documents,
            "chat_history": self.memory.buffer,
        }

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, object]]:
        if not question.strip():
            return []

        docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(question, k=top_k)
        retrieved = []
        for doc, score in docs_and_scores:
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page") or metadata.get("page_number")
            chunk_index = metadata.get("chunk_index")
            retrieved.append(
                {
                    "source": source,
                    "score": float(score),
                    "content": doc.page_content,
                    "page": page,
                    "chunk_index": chunk_index,
                }
            )
        return retrieved

    def reset(self) -> None:
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        self.chain.memory = self.memory

    def save_feedback(self, question: str, answer: str, rating: str, comment: Optional[str] = None) -> None:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().isoformat()
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp}\t{rating}\t{question}\t{answer}\t{comment or ''}\n")

    @staticmethod
    def _extract_sources(source_documents: List) -> Set[str]:
        sources: Set[str] = set()
        for doc in source_documents:
            metadata = getattr(doc, "metadata", {})
            sources.add(metadata.get("source", "unknown"))
        return sources


if __name__ == "__main__":
    assistant = HealthcareRAGAssistant()
    print("Healthcare RAG chatbot ready. Ask a question or type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if question.lower() == "reset":
            assistant.reset()
            print("Conversation history cleared.")
            continue
        response = assistant.chat(question)
        print(f"\nAssistant: {response['answer']}\n")
