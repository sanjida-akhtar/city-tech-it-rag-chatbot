# import libraries
from dotenv import load_dotenv
from data_loader import load_all_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
documents = load_all_documents("../data/raw/docs.json")


# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss", embeddings=embeddings, allow_dangerous_deserialization=True)

# retrieval

query = "What is city tech it?"
docs = vector_store.similarity_search(
    query,
    k=5
)

print(docs)