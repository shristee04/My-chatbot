import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os 
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.header("CHATBOT 📚")

with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload a PDF file and start asking question", type=["pdf"]) #file upload
if file is not None:
    if st.session_state.vector_store is None:
        with st.spinner("Reading your PDF... please wait ⏳"):
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() #extracting text

            #split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n"],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len)
            chunks = text_splitter.split_text(text)

            #convert into vectors(embeding)    
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings) #stored in FAISS database

        st.success("PDF ready! Ask your questions below ✅")

    #input qiestion from user
    user_question = st.text_input("Type your question here")

    if user_question:
        match = st.session_state.vector_store.similarity_search(user_question) #search FAISS for relevant chunks
        context = " ".join([doc.page_content for doc in match])

        st.session_state.messages.append({
        "role": "user", 
        "content": f"Answer this: {user_question}\n\nContext: {context}"
    })

        with st.spinner("Thinking... 🤔"):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages 
    )
        answer = response.choices[0].message.content
        st.write(answer)

        st.session_state.messages.append({
        "role": "assistant", 
        "content": answer
    })
        
else:
    st.info(" Please upload a PDF from the sidebar to get started!")
        