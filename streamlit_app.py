
# Set up ngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501, proto="http")
print(f"Streamlit app will be available on: {public_url}")
from pyngrok import ngrok

ngrok.set_auth_token("2kNaCcS4Z5JXTHUJic0LchbQPhP_3WkRdooqFqg3Ve1ZXnrYh")
# Streamlit app
import streamlit as st
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI

# Set up the language model
api_key = "AIzaSyCg3IaLdsrn4MhBSZ1PBJ7IxFq6j4T_xMM"
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.1)

# Initialize Spacy embeddings
embedding = SpacyEmbeddings(model_name="en_core_web_lg")

# Function to read PDF and create vector store
def setup_vector_store(pdf_path):
    doc = PdfReader(pdf_path)
    combined_text = ""
    for page in doc.pages:
        text = page.extract_text()
        if text:
            combined_text += text

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    data = text_splitter.split_text(combined_text)

    document_search = FAISS.from_texts(data, embedding)
    return document_search

# Load the question-answering chain
def setup_qa_chain():
    return load_qa_chain(llm, chain_type="stuff")

# Streamlit UI
def run_streamlit():
    st.title("PDF Question-Answering System")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("PDF file uploaded successfully.")

        # Save the uploaded file
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Set up vector store and QA chain
        document_search = setup_vector_store("uploaded_pdf.pdf")
        chain = setup_qa_chain()

        query = st.text_input("Enter your query")
        if query:
            docs = document_search.similarity_search(query)
            answer = chain.run(input_documents=docs, question=query)
            st.write("Answer:")
            st.write(answer)

# Run Streamlit in the background
import os
os.system("streamlit run app.py &")

# Use the public ngrok URL to access the app
print(f"Open the app in your browser: {public_url}")