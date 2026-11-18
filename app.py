import argparse
from typing import List, Dict
from chatbot import HealthcareRAGAssistant
from ingest import build_vector_store, VECTOR_STORE_DIR


def format_retrieval_details(retrieval_items: List[Dict]) -> str:
    """Format retrieval items for display."""
    if not retrieval_items:
        return "No retrieval details available."
    
    lines = ["📚 RETRIEVAL DETAILS:"]
    lines.append("─" * 80)
    scores = [item["score"] for item in retrieval_items]
    sources = {item["source"] for item in retrieval_items}
    
    lines.append(f"  ✓ Top Relevance Score:      {max(scores):.4f}")
    lines.append(f"  ✓ Average Relevance Score:  {sum(scores) / len(scores):.4f}")
    lines.append(f"  ✓ Chunks Retrieved:         {len(retrieval_items)}")
    lines.append(f"  ✓ Distinct Sources:         {len(sources)}")
    lines.append("\n  RETRIEVED PASSAGES:")
    lines.append("─" * 80)
    
    for idx, item in enumerate(retrieval_items, start=1):
        source = item["source"]
        score = item["score"]
        content = item["content"]
        location = []
        if item.get("page") is not None:
            location.append(f"page {item['page']}")
        if item.get("chunk_index") is not None:
            location.append(f"chunk {item['chunk_index']}")
        location_str = f" [{', '.join(location)}]" if location else ""
        
        lines.append(f"\n  [{idx}] SOURCE: {source}{location_str}")
        lines.append(f"      RELEVANCE SCORE: {score:.4f} {'⭐' * int(score * 5)}")
        lines.append(f"      CONTENT:")
        # Wrap content at 75 characters for readability
        for line in content.split('\n'):
            if line:
                # Split long lines
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 75:
                        current_line += word + " "
                    else:
                        if current_line:
                            lines.append(f"      {current_line.strip()}")
                        current_line = word + " "
                if current_line:
                    lines.append(f"      {current_line.strip()}")
            else:
                lines.append("")
    
    lines.append("\n" + "─" * 80)
    return "\n".join(lines)


def format_chat_history(chat_history) -> str:
    """Format chat history for display."""
    if not chat_history:
        return "No chat history yet."
    
    lines = ["📝 CONVERSATION HISTORY:"]
    for msg in chat_history:
        if hasattr(msg, 'type'):
            role = msg.type.upper() if hasattr(msg, 'type') else "MESSAGE"
        else:
            role = "MESSAGE"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        lines.append(f"  [{role}]: {content[:80]}{'...' if len(str(content)) > 80 else ''}")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare RAG conversational assistant.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the local FAISS vector store from docs.")
    parser.add_argument("--chat", action="store_true", help="Start a conversational chatbot.")
    parser.add_argument("--question", type=str, help="Ask a single question and exit.")
    args = parser.parse_args()

    if args.reindex or not VECTOR_STORE_DIR.exists():
        build_vector_store()

    if args.question:
        assistant = HealthcareRAGAssistant()
        response = assistant.chat(args.question)
        print(response["answer"])
    elif args.chat:
        assistant = HealthcareRAGAssistant()
        print("Healthcare RAG chatbot ready. Ask a question or type 'exit' to quit.")
        print("Type 'reset' to clear conversation history.\n")
        while True:
            question = input("\n📤 You: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break
            if question.lower() == "reset":
                assistant.reset()
                print("✅ Conversation history cleared.")
                continue
            
            # Get response and retrieval details
            response = assistant.chat(question)
            retrieval_details = assistant.retrieve(question)
            
            # Display answer
            print(f"\n🤖 Assistant: {response['answer']}")
            
            # Display retrieval details
            print(f"\n{format_retrieval_details(retrieval_details)}")
            
            # Display chat history
            print(f"\n{format_chat_history(response['chat_history'])}")
            print("\n" + "="*80)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
