# import os
# import uuid  
# import time  
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_pinecone import Pinecone
# from langchain_classic.vectorstores import pinecone
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import pinecone

# load_dotenv()

# # Custom CSS for professional styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#         padding-bottom: 0.5rem;
#         border-bottom: 2px solid #1f77b4;
#     }
#     .sidebar-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #1f77b4;
#         margin-bottom: 1rem;
#     }
#     .success-box {
#         background-color: #d4edda;
#         border: 1px solid #c3e6cb;
#         border-radius: 8px;
#         padding: 12px;
#         color: #155724;
#         margin: 10px 0;
#     }
#     .info-box {
#         background-color: #d1ecf1;
#         border: 1px solid #bee5eb;
#         border-radius: 8px;
#         padding: 12px;
#         color: #0c5460;
#         margin: 10px 0;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         border: 1px solid #ffeaa7;
#         border-radius: 8px;
#         padding: 12px;
#         color: #856404;
#         margin: 10px 0;
#     }
#     .chat-user {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border-radius: 15px 15px 5px 15px;
#         padding: 12px 16px;
#         margin: 8px 0;
#         margin-left: 20%;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .chat-assistant {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         border-radius: 15px 15px 15px 5px;
#         padding: 12px 16px;
#         margin: 8px 0;
#         margin-right: 20%;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .stButton button {
#         width: 100%;
#         border-radius: 8px;
#         font-weight: 600;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 10px;
#         transition: all 0.3s ease;
#     }
#     .stButton button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#     }
#     .uploaded-file {
#         background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
#         border: 2px dashed #667eea;
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         margin: 10px 0;
#         color: #333;
#     }
#     .provider-card {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#         text-align: center;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     .metric-card {
#         background: white;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 5px 0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         border-left: 4px solid #667eea;
#     }
#     /* Hide Streamlit default elements */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)

# DEFAULT_USER_PROMPT = "You are a helpful assistant. Your task is to answer the user's question based only on the following context. If the answer is not in the context, say 'I do not have enough information to answer that question.'"
# CONTEXT_SUFFIX = "\n\nContext:\n{context}"

# def sanitize_filename_for_pinecone(filename: str) -> str:
#     name_without_ext = os.path.splitext(filename)[0]
#     lower_name = name_without_ext.lower()
#     return lower_name

# def initialize_huggingface_components(api_key):
#     embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2", huggingfacehub_api_token=api_key)
#     repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
#     llm = HuggingFaceEndpoint(
#         repo_id=repo_id, task="text-generation", temperature=0.5,
#         huggingfacehub_api_token=api_key, max_new_tokens=512
#     )
    
#     return embeddings, ChatHuggingFace(llm=llm)

# def initialize_azure_openai_components(endpoint, api_key, deployment_name):
#     embeddings = AzureOpenAIEmbeddings(
#     deployment="text-embedding-3-small",
#     model="text-embedding-3-small",
#     openai_api_type="azure",
#     openai_api_key=api_key,
#     azure_endpoint=endpoint,
#     openai_api_version="2023-05-15",
#     chunk_size=2048
# )
    
#     chat_model = AzureChatOpenAI(azure_deployment=deployment_name, temperature=0.5, api_key=api_key, api_version='2023-03-15-preview', azure_endpoint=endpoint)
    
#     return embeddings, chat_model

# def initialize_google_genai_components(api_key):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#     chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=api_key)
    
#     return embeddings, chat_model

# def setup_pinecone_index(api_key, index_name, dimension):
#     # Initialize Pinecone client
#     pc = pinecone.Pinecone(api_key=api_key)
    
#     # Check if index exists, create if it doesn't
#     if index_name not in [index.name for index in pc.list_indexes()]:
#         st.info(f"Creating new Pinecone index: '{index_name}'")
#         pc.create_index(
#             name=index_name, 
#             dimension=dimension,
#             metric="cosine",
#             spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#         # Wait for index to be ready
#         while not pc.describe_index(index_name).status.ready:
#             time.sleep(1)
        
#     return pc.Index(index_name)

# def process_and_store_pdf(uploaded_file, text_splitter, embeddings, index_name):
#     temp_file_path = f"./{uploaded_file.name}"
    
#     with open(temp_file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
        
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()
    
#     texts = text_splitter.split_documents(documents)
    
#     # Use the correct Pinecone vector store implementation
#     vector_store = Pinecone.from_documents(
#         documents=texts,
#         embedding=embeddings,
#         index_name=index_name
#     )
    
#     os.remove(temp_file_path)
    
#     return len(texts), vector_store

# def ask_document(question, vector_store, chat_model, user_prompt):
#     retriever = vector_store.as_retriever()
    
#     retrieved_docs = retriever.invoke(question)
    
#     context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
#     full_prompt_template = user_prompt + CONTEXT_SUFFIX
#     final_system_prompt = full_prompt_template.format(context=context)
    
#     messages = [
#         SystemMessage(content=final_system_prompt),
#         HumanMessage(content=question),
#     ]
    
#     return chat_model.invoke(messages).content

# # Main App Layout
# st.markdown('<h1 class="main-header">📚 Document Intelligence Assistant</h1>', unsafe_allow_html=True)

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "rag_components" not in st.session_state:
#     st.session_state.rag_components = {}

# if "user_prompt" not in st.session_state:
#     st.session_state.user_prompt = DEFAULT_USER_PROMPT

# # Sidebar with improved layout
# with st.sidebar:
#     st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
#     # Mode selection
#     mode = st.radio(
#         "Select Mode",
#         ["📄 Document Q&A (RAG)", "💬 Direct Chat"],
#         help="Choose whether to chat with your documents or directly with the AI"
#     )
    
#     rag_on = (mode == "📄 Document Q&A (RAG)")
    
#     if rag_on:
#         st.markdown('<div class="sidebar-header">🔧 RAG Settings</div>', unsafe_allow_html=True)
        
#         # Provider selection with cards
#         st.markdown("**Embedding Model Provider**")
#         embedding_model_provider = st.selectbox(
#             "Select Provider",
#             ["Hugging Face", "Azure OpenAI", "Google Gemini"],
#             label_visibility="collapsed"
#         )
        
#         # Display provider card
#         provider_icons = {
#             "Hugging Face": "🤗",
#             "Azure OpenAI": "🔷", 
#             "Google Gemini": "🔮"
#         }
        
#         st.markdown(f"""
#         <div class="provider-card">
#             <strong>{provider_icons[embedding_model_provider]} {embedding_model_provider}</strong>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # File upload with styled container
#         st.markdown("**Document Upload**")
#         uploaded_file = st.file_uploader(
#             "Choose a PDF file", 
#             type="pdf",
#             label_visibility="collapsed",
#             help="Upload a PDF document to create a knowledge base"
#         )
        
#         if uploaded_file:
#             st.markdown(f"""
#             <div class="uploaded-file">
#                 📄 <strong>{uploaded_file.name}</strong><br>
#                 <small>Size: {uploaded_file.size / 1024:.1f} KB</small>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Assistant instructions with expander
#         with st.expander("🧠 Assistant Instructions", expanded=False):
#             st.session_state.user_prompt = st.text_area(
#                 "Customize how the assistant should respond:",
#                 value=st.session_state.user_prompt,
#                 height=150,
#                 label_visibility="collapsed"
#             )
            
#     else:
#         st.markdown('<div class="sidebar-header">💬 Chat Settings</div>', unsafe_allow_html=True)
#         llm_provider = st.selectbox("Select LLM Provider", ["Hugging Face", "Azure OpenAI", "Google Gemini"])

# # Main content area - simplified without tabs
# # Display chat messages with custom styling
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.markdown(f'<div class="chat-user"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<div class="chat-assistant"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

# # Status panel
# st.sidebar.markdown("### 📊 Status Panel")
    
# if rag_on:
#     if uploaded_file is not None:
#         st.sidebar.markdown('<div class="info-box">📄 Document Ready for Q&A</div>', unsafe_allow_html=True)
        
#         # Document info
#         st.sidebar.markdown(f"""
#         <div class="metric-card">
#             <strong>Document Name</strong><br>
#             {uploaded_file.name}
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.sidebar.markdown(f"""
#         <div class="metric-card">
#             <strong>File Size</strong><br>
#             {uploaded_file.size / 1024:.1f} KB
#         </div>
#         """, unsafe_allow_html=True)
        
#         if st.session_state.rag_components:
#             st.sidebar.markdown('<div class="success-box">✅ Knowledge Base Connected</div>', unsafe_allow_html=True)
#             st.sidebar.markdown(f"""
#             <div class="metric-card">
#                 <strong>Index Status</strong><br>
#                 🟢 Active
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.sidebar.markdown('<div class="warning-box">⏳ Processing Required</div>', unsafe_allow_html=True)
#     else:
#         st.sidebar.markdown('<div class="warning-box">📝 Please upload a document</div>', unsafe_allow_html=True)
# else:
#     st.sidebar.markdown('<div class="info-box">💬 Direct Chat Mode</div>', unsafe_allow_html=True)
#     st.sidebar.markdown(f"""
#     <div class="metric-card">
#         <strong>Provider</strong><br>
#         {llm_provider}
#     </div>
#     """, unsafe_allow_html=True)

# # Document processing logic
# if rag_on and uploaded_file is not None:
#     index_name = sanitize_filename_for_pinecone(uploaded_file.name)
    
#     if st.session_state.rag_components.get("index_name") != index_name:
#         with st.spinner(f"🔍 Connecting to knowledge base for '{uploaded_file.name}'..."):
#             try:
#                 if embedding_model_provider == "Hugging Face":
#                     embeddings, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
#                     dimension = 768
#                 elif embedding_model_provider == "Azure OpenAI":
#                     embeddings, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
#                     dimension = 1536
#                 else:  # Google Gemini
#                     embeddings, chat_model = initialize_google_genai_components(os.getenv("GOOGLE_API_KEY"))
#                     dimension = 768
                
#                 # Initialize Pinecone client
#                 pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#                 existing_indexes = [index.name for index in pc.list_indexes()]
                
#                 # Check if document-specific index exists, if not use default "pdf-data"
#                 if index_name not in existing_indexes:
#                     st.warning(f"Index '{index_name}' not found. Using default index 'pdf-data'.")
                    
#                     # Check if default index exists
#                     if "pdf-data" not in existing_indexes:
#                         st.info("Creating default index 'pdf-data'...")
#                         pc.create_index(
#                             name="pdf-data", 
#                             dimension=dimension,
#                             metric="cosine",
#                             spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
#                         )
#                         # Wait for index to be ready
#                         while not pc.describe_index("pdf-data").status.ready:
#                             time.sleep(1)
#                         st.success("Default index 'pdf-data' created successfully!")
                    
#                     # Use the default index
#                     index_name = "pdf-data"
                
#                 needs_processing = index_name not in st.session_state.rag_components

#                 if needs_processing:
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#                     with st.spinner(f"📄 Processing '{uploaded_file.name}'..."):
#                         num_chunks, vector_store = process_and_store_pdf(uploaded_file, text_splitter, embeddings, index_name)
                    
#                     # Success message with metrics
#                     st.success("✅ Document Processing Complete!")
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.markdown(f"""
#                         <div class="metric-card">
#                             <strong>Document Chunks</strong><br>
#                             <h3>{num_chunks}</h3>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     with col2:
#                         st.markdown(f"""
#                         <div class="metric-card">
#                             <strong>Index Name</strong><br>
#                             {index_name}
#                         </div>
#                         """, unsafe_allow_html=True)
#                     with col3:
#                         st.markdown(f"""
#                         <div class="metric-card">
#                             <strong>Status</strong><br>
#                             🟢 Ready
#                         </div>
#                         """, unsafe_allow_html=True)
#                 else:
#                     # Connect to existing index
#                     vector_store = Pinecone.from_existing_index(
#                         index_name=index_name,
#                         embedding=embeddings
#                     )
#                     st.success(f"✅ Connected to existing knowledge base for '{uploaded_file.name}'.")
                
#                 st.session_state.rag_components = {
#                     "vector_store": vector_store,
#                     "chat_model": chat_model,
#                     "index_name": index_name
#                 }
                
#                 # Refresh the app to show updated status
#                 st.rerun()
                
#             except Exception as e:
#                 st.error(f"❌ An error occurred: {e}")
#                 st.session_state.rag_components = {}

# # Chat input at bottom
# st.markdown("---")
# input_col, button_col = st.columns([4, 1])

# with input_col:
#     prompt = st.chat_input("💬 Ask your question here...")

# with button_col:
#     if st.session_state.messages:
#         if st.button("🗑️ Clear", use_container_width=True):
#             st.session_state.messages = []
#             st.rerun()

# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message immediately
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate and display assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("🤔 Thinking..."):
#             try:
#                 if rag_on:
#                     if st.session_state.rag_components:
#                         vs = st.session_state.rag_components["vector_store"]
#                         cm = st.session_state.rag_components["chat_model"]
#                         answer = ask_document(prompt, vs, cm, st.session_state.user_prompt)
#                     else:
#                         answer = "📝 Please upload and process a document to begin the RAG chat."
#                 else:
#                     if llm_provider == "Hugging Face":
#                         _, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
#                     elif llm_provider == "Azure OpenAI":
#                         _, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
#                     else:  # Google Gemini
#                         _, chat_model = initialize_google_genai_components(os.getenv("GOOGLE_API_KEY"))
                    
#                     answer = chat_model.invoke([HumanMessage(content=prompt)]).content
                
#                 st.markdown(answer)
#                 st.session_state.messages.append({"role": "assistant", "content": answer})
                
#             except Exception as e:
#                 st.error(f" An error occurred: {e}")


import os
import uuid  
import time  
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from langchain_classic.vectorstores import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pinecone

load_dotenv()

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 12px;
        color: #155724;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 12px;
        color: #0c5460;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 12px;
        color: #856404;
        margin: 10px 0;
    }
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px 15px 5px 15px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px 15px 15px 5px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .uploaded-file {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        color: #333;
    }
    .provider-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DEFAULT_USER_PROMPT = "You are a helpful assistant. Your task is to answer the user's question based only on the following context. If the answer is not in the context, say 'I do not have enough information to answer that question.'"
CONTEXT_SUFFIX = "\n\nContext:\n{context}"

def sanitize_filename_for_pinecone(filename: str) -> str:
    name_without_ext = os.path.splitext(filename)[0]
    lower_name = name_without_ext.lower()
    return lower_name

def initialize_huggingface_components(api_key):
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2", huggingfacehub_api_token=api_key)
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, task="text-generation", temperature=0.5,
        huggingfacehub_api_token=api_key, max_new_tokens=512
    )
    
    return embeddings, ChatHuggingFace(llm=llm)

def initialize_azure_openai_components(endpoint, api_key, deployment_name):
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        openai_api_type="azure",
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        openai_api_version="2023-05-15",
        chunk_size=2048
    )
    
    chat_model = AzureChatOpenAI(azure_deployment=deployment_name, temperature=0.5, api_key=api_key, api_version='2023-03-15-preview', azure_endpoint=endpoint)
    
    return embeddings, chat_model

def initialize_google_genai_components(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=api_key)
    
    return embeddings, chat_model

def setup_pinecone_index(api_key, index_name, dimension):
    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=api_key)
    
    # Check if index exists, create if it doesn't
    if index_name not in [index.name for index in pc.list_indexes()]:
        st.info(f"Creating new Pinecone index: '{index_name}'")
        pc.create_index(
            name=index_name, 
            dimension=dimension,
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status.ready:
            time.sleep(1)
        
    return pc.Index(index_name)

def process_and_store_pdf(uploaded_file, text_splitter, embeddings, index_name):
    temp_file_path = f"./{uploaded_file.name}"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    texts = text_splitter.split_documents(documents)
    
    # Use the correct Pinecone vector store implementation
    vector_store = Pinecone.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=index_name
    )
    
    os.remove(temp_file_path)
    
    return len(texts), vector_store

def ask_document(question, vector_store, chat_model, user_prompt):
    retriever = vector_store.as_retriever()
    
    retrieved_docs = retriever.invoke(question)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    full_prompt_template = user_prompt + CONTEXT_SUFFIX
    final_system_prompt = full_prompt_template.format(context=context)
    
    messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=question),
    ]
    
    return chat_model.invoke(messages).content

# Main App Layout
st.markdown('<h1 class="main-header">📚 Document Intelligence Assistant</h1>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_components" not in st.session_state:
    st.session_state.rag_components = {}

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_USER_PROMPT

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar with improved layout
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["📄 Document Q&A (RAG)", "💬 Direct Chat"],
        help="Choose whether to chat with your documents or directly with the AI"
    )
    
    rag_on = (mode == "📄 Document Q&A (RAG)")
    
    if rag_on:
        st.markdown('<div class="sidebar-header">🔧 RAG Settings</div>', unsafe_allow_html=True)
        
        # Provider selection with cards
        st.markdown("**Embedding Model Provider**")
        embedding_model_provider = st.selectbox(
            "Select Provider",
            ["Hugging Face", "Azure OpenAI", "Google Gemini"],
            label_visibility="collapsed"
        )
        
        # Display provider card
        provider_icons = {
            "Hugging Face": "🤗",
            "Azure OpenAI": "🔷", 
            "Google Gemini": "🔮"
        }
        
        st.markdown(f"""
        <div class="provider-card">
            <strong>{provider_icons[embedding_model_provider]} {embedding_model_provider}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload with styled container
        st.markdown("**Document Upload**")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            label_visibility="collapsed",
            help="Upload a PDF document to create a knowledge base"
        )
        
        if uploaded_file:
            st.markdown(f"""
            <div class="uploaded-file">
                📄 <strong>{uploaded_file.name}</strong><br>
                <small>Size: {uploaded_file.size / 1024:.1f} KB</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Process button - only show when a new file is uploaded
            if st.session_state.current_file != uploaded_file.name or not st.session_state.document_processed:
                if st.button("🚀 Process Document", use_container_width=True):
                    with st.spinner("🔍 Setting up document processing..."):
                        try:
                            index_name = sanitize_filename_for_pinecone(uploaded_file.name)
                            
                            if embedding_model_provider == "Hugging Face":
                                embeddings, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
                                dimension = 768
                            elif embedding_model_provider == "Azure OpenAI":
                                embeddings, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
                                dimension = 1536
                            else:  # Google Gemini
                                embeddings, chat_model = initialize_google_genai_components(os.getenv("GOOGLE_API_KEY"))
                                dimension = 768
                            
                            # Initialize Pinecone client
                            pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                            existing_indexes = [index.name for index in pc.list_indexes()]
                            
                            # Check if document-specific index exists, if not use default "pdf-data"
                            if index_name not in existing_indexes:
                                st.warning(f"Index '{index_name}' not found. Using default index 'pdf-data'.")
                                
                                # Check if default index exists
                                if "pdf-data" not in existing_indexes:
                                    st.info("Creating default index 'pdf-data'...")
                                    pc.create_index(
                                        name="pdf-data", 
                                        dimension=dimension,
                                        metric="cosine",
                                        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
                                    )
                                    # Wait for index to be ready
                                    while not pc.describe_index("pdf-data").status.ready:
                                        time.sleep(1)
                                    st.success("Default index 'pdf-data' created successfully!")
                                
                                # Use the default index
                                index_name = "pdf-data"
                            
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            with st.spinner(f"📄 Processing '{uploaded_file.name}'..."):
                                num_chunks, vector_store = process_and_store_pdf(uploaded_file, text_splitter, embeddings, index_name)
                            
                            # Store in session state
                            st.session_state.rag_components = {
                                "vector_store": vector_store,
                                "chat_model": chat_model,
                                "index_name": index_name
                            }
                            st.session_state.document_processed = True
                            st.session_state.current_file = uploaded_file.name
                            
                            # Success message with metrics
                            st.success("✅ Document Processing Complete!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ An error occurred during processing: {e}")
        
        # Assistant instructions with expander
        with st.expander("🧠 Assistant Instructions", expanded=False):
            st.session_state.user_prompt = st.text_area(
                "Customize how the assistant should respond:",
                value=st.session_state.user_prompt,
                height=150,
                label_visibility="collapsed"
            )
            
    else:
        st.markdown('<div class="sidebar-header">💬 Chat Settings</div>', unsafe_allow_html=True)
        llm_provider = st.selectbox("Select LLM Provider", ["Hugging Face", "Azure OpenAI", "Google Gemini"])

# Main content area - Display chat messages with custom styling
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-user"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

# Status panel
st.sidebar.markdown("### 📊 Status Panel")
    
if rag_on:
    if uploaded_file is not None:
        st.sidebar.markdown('<div class="info-box">📄 Document Ready for Q&A</div>', unsafe_allow_html=True)
        
        # Document info
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <strong>Document Name</strong><br>
            {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <strong>File Size</strong><br>
            {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.document_processed and st.session_state.rag_components:
            st.sidebar.markdown('<div class="success-box">✅ Knowledge Base Connected</div>', unsafe_allow_html=True)
            st.sidebar.markdown(f"""
            <div class="metric-card">
                <strong>Index Status</strong><br>
                🟢 Active
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="warning-box">⏳ Processing Required</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="warning-box">📝 Please upload a document</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="info-box">💬 Direct Chat Mode</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Provider</strong><br>
        {llm_provider}
    </div>
    """, unsafe_allow_html=True)

# Chat input at bottom
st.markdown("---")
input_col, button_col = st.columns([4, 1])

with input_col:
    prompt = st.chat_input("💬 Ask your question here...")

with button_col:
    if st.session_state.messages:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                if rag_on:
                    if st.session_state.document_processed and st.session_state.rag_components:
                        vs = st.session_state.rag_components["vector_store"]
                        cm = st.session_state.rag_components["chat_model"]
                        answer = ask_document(prompt, vs, cm, st.session_state.user_prompt)
                    else:
                        answer = "📝 Please upload and process a document to begin the RAG chat. Click the 'Process Document' button after uploading your PDF."
                else:
                    if llm_provider == "Hugging Face":
                        _, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
                    elif llm_provider == "Azure OpenAI":
                        _, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
                    else:  # Google Gemini
                        _, chat_model = initialize_google_genai_components(os.getenv("GOOGLE_API_KEY"))
                    
                    answer = chat_model.invoke([HumanMessage(content=prompt)]).content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f" An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})