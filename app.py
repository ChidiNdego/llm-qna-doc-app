# Import necessary libraries
import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import tiktoken
import os


# Load and process documents based on their format
def load_document(file):
    _, extension = os.path.splitext(file)
    loader = None

    # Determine the file type and select the appropriate loader
    if extension.lower() == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader # prevent similar dependencies
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension.lower() == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension.lower() == '.txt':
        from langchain_community.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file)
    else:
        print(f'Document format "{extension}" is not supported!')
        return None
    
    data = loader.load() # Load the document data
    return data

# Chunk the loaded document data for processing
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings for the chunked data
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Use the conversational retrieval chain to ask and get answers from the LLM
def ask_and_get_answer(vector_store, q, memory, k=3): # increase k for more elaborate value; however, more tokens, more cost
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k}) # 3 most similar answer to the user's query
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=retriever, memory=memory)    
    answer = chain.invoke(q)['answer']
    return answer

# Calculate the embedding cost of the processed texts
def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_token = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_token,  total_token/1000 * 0.0004

# Clear the session history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        del st.session_state['messages']
    return


# Main application setup and logic
if __name__=="__main__":
    load_dotenv(find_dotenv(), override=True)

    st.set_page_config(page_title='Your Custom Assistant', page_icon='🤖')

    # Initialize session state for messages and memory
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    st.subheader('LLM Question-Answering Application 🤖')

    # Sidebar setup for API key input and file uploading
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        # File processing and embedding upon upload
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking, and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name_without_extension = os.path.splitext(uploaded_file.name)[0]
                folder_name = os.path.join('./', file_name_without_extension)

                outer_folder = "uploaded_files"
                if not os.path.exists(outer_folder):
                    os.makedirs(outer_folder)

                folder_name = os.path.join('./', outer_folder, file_name_without_extension)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                file_path = os.path.join(folder_name, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(bytes_data)
                
                data = load_document(file_path)
                chunks = chunk_data(data, chunk_size=chunk_size)
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store

                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                st.success('File uploaded, chunked and embedded successfully.')
        
        # Developer credit
        my_name = "Chidi Ndego"
        linkedin_url = "https://www.linkedin.com/in/chidindego"
        st.markdown(f"Developed by <a href='{linkedin_url}'>{my_name}</a>", unsafe_allow_html=True)
    
    chat_container = st.container()
    input_container = st.container()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with input_container:                
        q = st.text_input('Ask a question about the content of your file:', key="question", value="")
        submit_button = st.button('Submit')

    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if submit_button and q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            memory = st.session_state['memory']
            answer = ask_and_get_answer(vector_store, q, memory, k)
         
            st.session_state.messages.append(q)
            st.session_state.messages.append(answer)

            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

    with chat_container:
        for i,msg in enumerate(st.session_state.messages):
            if i%2==0:
                message(msg, is_user=True, key=f'{i} + 🥸')
            else:
                message(msg, is_user=False, key=f'{i} + 🤖')