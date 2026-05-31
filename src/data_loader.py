from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
import json


def load_all_documents(data_dir : str) -> List [Any]:
    """"load all json files and convert to document structure"""
    # use project root data folder

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] data path: {data_path}")

    with open (data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        doc = Document(
            page_content = item["content"],
            metadata = {"source" : item.get("source", "unknown")}

        )
        documents.append(doc)

    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("../data/raw/docs.json")
    print(f"[DEBUG] Loaded {len(docs)} documents")
    print("Example document: ", docs[0] if docs else None)
