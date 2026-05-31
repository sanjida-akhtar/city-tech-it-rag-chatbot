import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from data_loader import load_all_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

path = os.path.join(os.path.dirname(__file__), "city_tech_it_logo.jpg")
st.set_page_config(page_title = "City Tech IT AI Assistant", page_icon = path, layout = "wide")
# sidebar
with st.sidebar:
    st.image(path)

# main chat

st.title("City Tech IT AI Assistant")
st.caption("Ask anything about us!")

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss", embeddings=embeddings, allow_dangerous_deserialization=True)
llm = ChatGroq(api_key = os.environ.get("GROQ_API_KEY"), model = "llama-3.1-8b-instant")

output = StrOutputParser()


# create the bar where we can type messages
query = st.chat_input("How can I help you?")
# retrieval
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 2}
    )

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.

<context>
{context}
</context>

Question: {input}
""")

chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)

if query:
    with st.spinner("Thinking..."):
        response = chain.invoke(query)

        st.write(response.content)


