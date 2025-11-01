import os
from typing import List
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_classic import  conversational_retrieval
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
import tempfile

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Student Assistant Chatbot",
    page_icon="🎓",
    layout="wide"
)



AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_OPENAI_KEY=os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME=os.getenv("DEPLOYMENT_NAME")



# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation' not in st.session_state:
    st.session_state.conversation = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)



# Title
st.markdown("<h1 class='main-header'>🎓 Student Assistant Chatbot</h1>", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
  
    st.divider()
    
    st.header("📚 Data Sources")
    
    # Tab selection for data input
    data_source = st.radio(
        "Choose data source:",
        ["Upload PDF", "Scrape Website"]
    )
    
    if data_source == "Upload PDF":
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files containing college information"
        )
        
        if st.button("Process PDFs", type="primary"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    try:
                        documents = []
                        
                        for uploaded_file in uploaded_files:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_file_path = tmp_file.name
                            
                            # Load PDF
                            loader = PyPDFLoader(tmp_file_path)
                            docs = loader.load()
                            documents.extend(docs)
                            
                            # Clean up temp file
                            os.unlink(tmp_file_path)
                        
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        # Create embeddings
                        embeddings = AzureOpenAIEmbeddings(
                            deployment="text-embedding-3-small",
                            model="text-embedding-3-small",
                            openai_api_type="azure",
                            openai_api_key=AZURE_OPENAI_KEY,
                            azure_endpoint=AZURE_OPENAI_ENDPOINT,
                            openai_api_version="2023-05-15",
                            chunk_size=2048
                        )
                                        
                        # Create vector store
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings,
                            persist_directory="./chroma_db"
                        )
                        
                        st.success(f"✅ Processed {len(uploaded_files)} PDF(s) with {len(splits)} chunks!")
                        
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file")
    
    else:  # Scrape Website
        website_url = st.text_input(
            "College Website URL",
            placeholder="https://example.edu",
            help="Enter the URL of your college website"
        )
        
        if st.button("Scrape Website", type="primary"):
            if website_url:
                with st.spinner("Scraping website..."):
                    try:
                        # Load web content
                        loader = WebBaseLoader(website_url)
                        documents = loader.load()
                        
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        # Create embeddings
                        embeddings = AzureOpenAIEmbeddings(
                            deployment="text-embedding-3-small",
                            model="text-embedding-3-small",
                            openai_api_type="azure",
                            openai_api_key=AZURE_OPENAI_KEY,
                            azure_endpoint=AZURE_OPENAI_ENDPOINT,
                            openai_api_version="2023-05-15",
                            chunk_size=2048
                        )
                        
                        # Create vector store
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings,
                            persist_directory="./chroma_db"
                        )
                        
                        st.success(f"✅ Scraped website with {len(splits)} chunks!")
                        
                    except Exception as e:
                        st.error(f"Error scraping website: {str(e)}")
            else:
                st.warning("Please enter a valid URL")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.rerun()

# Main chat interface
st.header("💬 Chat with Your Student Assistant")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message assistant-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

# Chat input
user_question = st.chat_input("Ask me anything about your college...")

if user_question:
    if st.session_state.vectorstore is None:
        st.error("⚠️ Please upload a PDF or scrape a website first")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("Thinking..."):
            try:
                # Initialize conversation chain if not exists
                if st.session_state.conversation is None:
                    # Custom prompt template
                    prompt_template = """You are a helpful student assistant. Your role is to help students by answering their questions about college information, courses, admissions, facilities, and any other academic-related queries.

Use the following context to answer the student's question. If you don't know the answer based on the context, say so politely and offer to help with related information.

Context: {context}

Student Question: {question}

Helpful Answer:"""

                    PROMPT = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    # Create LLM
                    llm = AzureChatOpenAI(azure_deployment=DEPLOYMENT_NAME, temperature=0.5, api_key=AZURE_OPENAI_KEY, api_version='2023-03-15-preview', azure_endpoint=AZURE_OPENAI_ENDPOINT)

                    
                    # Create memory
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer"
                    )
                    
                    st.session_state.conversation = conversational_retrieval.from_llm(
                        llm=llm,
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        memory=memory,
                        combine_docs_chain_kwargs={"prompt": PROMPT},
                        return_source_documents=True
                    )
                
                # Get response
                response = st.session_state.conversation({
                    "question": user_question
                })
                
                assistant_response = response['answer']
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Getting Started:
    
    1. **Add Your API Key**: Enter your OpenAI API key in the sidebar
    
    2. **Choose Data Source**:
       - **Upload PDF**: Upload PDF files containing college information
       - **Scrape Website**: Enter your college website URL to scrape data
    
    3. **Process Data**: Click the process button to convert data into vector embeddings
    
    4. **Start Chatting**: Ask questions about your college in the chat box below
    
    ### Example Questions:
    - What courses are offered?
    - Tell me about admission requirements
    - What facilities are available?
    - What is the fee structure?
    - Information about scholarships
    
    ### Requirements:
    ```bash
    pip install streamlit langchain langchain-community chromadb
    pip install sentence-transformers pypdf openai
    pip install beautifulsoup4 tiktoken
    ```
    
    ### Run the Application:
    ```bash
    streamlit run app.py
    ```
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with LangChain 🦜🔗 and ChromaDB 💾</p>
</div>
""", unsafe_allow_html=True)