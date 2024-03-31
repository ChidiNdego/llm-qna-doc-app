# LLM Question-Answering Application 🤖

### Description
This project leverages Large Language Models (LLMs) to offer a sophisticated question-answering system built on Streamlit. It dynamically processes uploaded documents (.pdf, .docx, .txt), extracts content, and creates embeddings to serve as the basis for generating context-aware responses to user queries.

### Features
- Document Processing: Supports PDF, DOCX, and TXT files.
- Embedding Generation: Utilizes OpenAI Embeddings for document understanding.
- Conversational Memory: Maintains context through ConversationBufferMemory.
- Interactive UI: Built with Streamlit for a responsive, user-friendly experience.

### Setup & Installation

### How to Use
1. Start the Streamlit application

2. Input your OpenAI API key in the sidebar
3. Upload a document
4. Set the chunk size and number os responses, k (Optional)
5. Ask questions based on the document content in the provided text input field.