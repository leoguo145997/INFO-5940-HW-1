import streamlit as st
from openai import AzureOpenAI
from os import environ
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
#learned the method PyPDF2 to load pdf by ChatGPT
import PyPDF2
import io

st.title("📝 File Q&A with OpenAI")
uploaded_files = st.file_uploader("Upload articles", type=("txt", "pdf"), accept_multiple_files=True)

question = st.chat_input(
    "Ask something about the articles",
    disabled=not uploaded_files,
)

# Initialize LangChain's AzureOpenAI wrapper
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    temperature=0.2,
    api_version="2023-06-01-preview",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Upload documents and ask questions about them!"}]
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Upload documents and ask questions about them!"}]
if "file_contents" not in st.session_state:
    st.session_state["file_contents"] = {}

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["file_contents"]:
            try:
                if uploaded_file.type == "text/plain":
                    # Read the content of the uploaded file
                    file_content = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    file_content = ""
                    for page in pdf_reader.pages:
                        file_content += page.extract_text()
                else:
                    st.error(f"Unsupported file type for {uploaded_file.name}. Please upload .txt or .pdf files.")
                    continue
                
                document = Document(page_content=file_content, metadata={"source": uploaded_file.name})

                chunk_size = 1000
                chunk_overlap = 200

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = text_splitter.split_documents([document])
                
                st.session_state["file_contents"][uploaded_file.name] = "\n\n".join([chunk.page_content for chunk in chunks])
                
                st.success(f"Processed {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Prepare prompt
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Setup retrieval
if st.session_state["vectorstore"] is not None:
    retriever = st.session_state["vectorstore"].as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

if st.session_state["file_contents"]:
    st.write("Processed files:")
    for file in st.session_state["file_contents"].keys():
        st.write(f"- {file}")

if question and uploaded_files:
    client = AzureOpenAI(
        api_key=environ['AZURE_OPENAI_API_KEY'],
        api_version="2023-03-15-preview",
        azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
    )

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        all_contents = "\n\n".join([f"Content of {filename}:\n{content}" 
                                    for filename, content in st.session_state["file_contents"].items()])
        
        stream = client.chat.completions.create(
            model="gpt-4o",  # Change this to a valid model name
            messages=[
                {"role": "system", "content": f"Here's the content of the file:\n\n{all_contents}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})

