import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from io import StringIO

# Securely fetch OpenAI API Key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
if not OPENAI_API_KEY:
    st.error("OpenAI API Key is missing. Please add it to Streamlit secrets.")
    st.stop()

@st.cache_data
def load_document(file_content):
    """Load and process text document for embedding."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.create_documents([file_content])
        return docs
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

@st.cache_data
def create_vectorstore(docs):
    """Creates FAISS vector store from documents."""
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def build_rag_chain(vectorstore):
    """Create a RetrievalQA chain."""
    try:
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"Error building RAG chain: {e}")
        return None

# Streamlit UI
def main():
    st.title("📄 RAG Application with LangChain & Streamlit")

    # File upload
    uploaded_file = st.file_uploader("Upload a document (text format)", type=["txt"])
    
    if uploaded_file is not None:
        st.info("Processing document...")
        
        # Read file content
        file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

        # Process document
        docs = load_document(file_content)
        if not docs:
            st.error("Failed to process document.")
            return
        
        # Create vector store
        vector_store = create_vectorstore(docs)
        if not vector_store:
            st.error("Failed to create vector store.")
            return

        # Build RAG chain
        rag_chain = build_rag_chain(vector_store)
        if not rag_chain:
            st.error("Failed to initialize RAG system.")
            return

        st.success("Document processed successfully. RAG system is ready.")

        # Query input
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Retrieving and generating response..."):
                try:
                    answer = rag_chain.run(query)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()

