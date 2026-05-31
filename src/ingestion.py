# import libraries
from dotenv import load_dotenv
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from data_loader import load_all_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

documents = load_all_documents("../data/raw/docs.json")
# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#dim = 384
#index = faiss.IndexFlatL2(dim)
vector_store = FAISS.from_documents(documents, embedding=embeddings)


# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
documents = text_splitter.split_documents(documents)

# generate unique id's

i = 0
uuids = []

while i < len(documents):

    i += 1

    uuids.append(f"id{i}")

# add to database

vector_store.add_documents(documents=documents, ids=uuids)
vector_store.save_local("faiss")