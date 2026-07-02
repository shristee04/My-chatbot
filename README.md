# My PDF Chatbot 📚

A chatbot that lets you upload a PDF and ask questions about it using AI.

## Features
- Upload any PDF and ask questions
- Remembers previous questions in the conversation
- Powered by Groq + Llama 3.3 70B

## How to run

1. Clone the repo
2. Install dependencies
   pip install -r requirement.txt
3. Create a .env file and add your Groq API key
   GROQ_API_KEY=your_key_here
4. Run the app
   streamlit run chatbot.py

## Tech Used
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API