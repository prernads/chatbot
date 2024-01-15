import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai

OPENAI_API_KEY = "Use your API key"  #OpenAI key

#Upload PDF files
st.header('My First Chatbot')

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF and Start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
        #st.write(text)

#Break it into chunks
    text_splitter =  RecursiveCharacterTextSplitter(
        separators= "\n",
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )  
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #Generating Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    #Creating Vector Store - FAISS
    vector_store = faiss.FAISS.from_texts(chunks,embeddings)

    #Get User Question
    user_question = st.text_input("Type your question here") 

    #Do Similarity Search
    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)

        #Generate LLM
        llm = openai(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0, 
            max_tokens = 1000,
            model_name = "SmartAI"
        )

    #Output Results
    #chain -> take the question, get relevant document, pass it to LLM model, output
    chain = load_qa_chain(llm,chain_type="stuff") 
    response = chain.run(input_documents = match, question = user_question)
    st.write(response)                    
             