# (D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\venv) 
# D:\Udemy\Complete_GenAI_Langchain_Huggingface\UPractice2\RAG_QA_Chatbot>streamlit run 1-Streamlit_app.py
# Python == 3.10

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

## streamlit framework
st.set_page_config("🤖 Chatbot")

st.sidebar.title("Settings")
groq_api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")
engine=st.sidebar.selectbox("Select Open Source model",["openai/gpt-oss-120b","openai/gpt-oss-20b"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7) # value=0.7: default temperature
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150) # value=150: default max_tokens

st.header("🌐 RAG Q&A Chatbot With Groq")
st.subheader("🖊️ What can I help with?", divider="grey")
file = st.file_uploader("Enter the PDF File", type="pdf", accept_multiple_files=False)
input_text=st.chat_input("Ask anything from the uploaded file...")

if not groq_api_key and not file:
    st.warning("Please enter the Groq API Key & upload the PDF file before you proceed...")

if st.button("Document Embedding") and groq_api_key and file:
    with st.spinner("Learning the Document..."):
        if "db_retriever" not in st.session_state:
            with open("temporary.pdf", "wb") as f:
                f.write(file.read())
            file_load = PyPDFLoader("./temporary.pdf").load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=50)
            file_chunk = text_splitter.split_documents(file_load)
            embeddings = HuggingFaceEmbeddings()
            db = FAISS.from_documents(file_chunk,embeddings)
            st.session_state.db_retriever = db.as_retriever()
    st.success("Learned the document. Now you can ask any question.")

if "db_retriever" in st.session_state and input_text:
    # llm, prompt,output_parser,chain
    llm = ChatGroq(api_key=groq_api_key,model=engine)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Given the recent user question and also the chat history, "
        "generate answer of the question from the retiever. Generate the answer effenciently "
        "from the given Context: {context}. This is the Context. Now generate anster from this retriever-context."),("user","{input}")])
    output_parser = StrOutputParser()

    chain = create_stuff_documents_chain(prompt=prompt, llm=llm, output_parser=output_parser)
    rag_chain = create_retrieval_chain(st.session_state.db_retriever, chain)

    # Question
    response = rag_chain.invoke({"input":input_text})
    st.write("Your Question:", input_text)
    st.success(response["answer"])
    st.write("Thank You. I hope it helped. Dont hesitate to ask the next question. 😊")
    st.warning("[[ If you want me to learn any Other Document, please refresh the page and upload the Document entering the API Key. Otherwise, no need to upload the same file again ]]")