import streamlit as st
from langchain_openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import tempfile
from langchain_community.document_loaders import PyPDFLoader

def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is not None:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # The rest of the workflow (splitting, embeddings, and QA) remains the same
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents) # Note: use split_documents here
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 2})
        
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key),
            chain_type='stuff', 
            retriever=retriever
        )
        response = qa.invoke({"query": query_text})
        return response["result"]

# Page title configuration
st.set_page_config(page_title='Ask the Doc App')
st.title('Ask the Doc App')

# File upload widget
uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf'])

# Query text input - disabled until a file is uploaded
query_text = st.text_input(
    'Enter your question:', 
    placeholder='Please provide a short summary.', 
    disabled=not uploaded_file
)

# Form input and query logic
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input(
        'OpenAI API Key', 
        type='password', 
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        'Submit', 
        disabled=not (uploaded_file and query_text)
    )
    
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

# Display the result
if len(result):
    st.info(response)