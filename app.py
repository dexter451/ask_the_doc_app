import streamlit as st
import tempfile
import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Page Configuration
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Handle PDF vs Text
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            
            # Select embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Create FAISS vector store (More stable for Streamlit Cloud than Chroma)
            db = FAISS.from_documents(texts, embeddings)
            
            # Create retriever with k=2 to avoid token limit errors
            retriever = db.as_retriever(search_kwargs={"k": 2})
            
            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=openai_api_key),
                chain_type='stuff', 
                retriever=retriever
            )
            
            response = qa.invoke({"query": query_text})
            return response["result"]

        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# App UI
uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf'])
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form to accept API Key and Question
with st.form('myform'):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            res = generate_response(uploaded_file, openai_api_key, query_text)
            st.info(res)
    elif submitted and not openai_api_key.startswith('sk-'):
        st.warning('Please enter a valid OpenAI API key!', icon='⚠️')