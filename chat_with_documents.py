import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import os
from streamlit_chat import message


def load_document(file):
    _, extension = os.path.splitext(file)

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

    data = loader.load() # return list of langchain doc. One doc for each page
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, memory, k=3): # increase k for more elaborate value; however, more tokens, more cost
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k}) # 3 most similar answer to the user's query
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        memory=memory
    )
    
    answer = chain.invoke(q)['answer']
    return answer


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_token = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_token,  total_token/1000 * 0.0004


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        del st.session_state['messages']
    return


if __name__=="__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.set_page_config(
        page_title='Your Custom Assistant',
        page_icon='🤖'
    )

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # st.image('84555.jpg')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking, and embedding file ...'):
                # Read the uploaded file's bytes
                bytes_data = uploaded_file.read()

                # Extract the file name without the extension
                file_name_without_extension = os.path.splitext(uploaded_file.name)[0]

                # Create a folder name based on the file name (without the extension)
                folder_name = os.path.join('./', file_name_without_extension)

                # Create the folder if it doesn't already exist
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Construct the file path including the new folder
                file_path = os.path.join(folder_name, uploaded_file.name)

                # Write the bytes data to the file within the new folder
                with open(file_path, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_path)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    input_container = st.container()

    with input_container:                
        q = st.text_input('Ask a question about the content of your file:', key="question", value="")
        submit_button = st.button('Submit')

    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

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