# INFO-5940-HW-1
Develop a Retrieval Augmented Generation (RAG) application that enables users to upload documents and interact with their content through a conversational interface. The application should efficiently handle large files by breaking them into smaller chunks and support multiple file formats and multiple document uploads.

Feature:
1. Multi-document upload support (.txt and .pdf files)
2. Document processing and chunking for efficient retrieval
3. Conversational AI interface for asking questions about uploaded documents
4. Ability to reference information from multiple documents in a single query

To upload multiple documents, you should:
1. Locate the "Upload articles" section near the top of the page.
2. Click on the "Browse files" button or drag and drop your files into the designated area.
3. You can upload multiple files at once. Supported file types are .txt and .pdf.
4. As each file is processed, you'll see a success message: "Processed [filename]".
5. After uploading, you'll see a list of "Processed files:" below the upload area.

To ask questions to the chatbot, you should:
1. Scroll down to find the chat input box at the bottom of the page.
2. Type your question about the uploaded documents into this box.
3. Press Enter or click the send button to submit your question.
4. You can ask multiple questions in succession, creating a conversation thread.

What I updated over the original configuration:
1. Extended file support: Added PDF handling alongside text files.
2. Multiple file uploads: Allows users to upload and process multiple documents.
3. Document processing: Implemented document chunking to better handling of large texts.
4. Langchain integration: Utilized Langchain's document handling capabilities, setting up for potential future enhancements.
   
Configuration Changes
I updated the Python version in the Dockerfile to 3.8 for compatibility with all required libraries and added the following packages to the requirements.txt file:
1. streamlit
2. openai
3. PyPDF2
4. langchain
5. langchain_openai
6. langchain_community
7. chromadb
