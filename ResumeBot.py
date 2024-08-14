import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
warnings.filterwarnings("ignore")
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
You are Kaushikee's intelligent assistant with a vast knowledge base. 
Your task is to read the given resume and answer the question asked based on the information provided in the resume. 
While answering the question please adhere to the following guidelines:

1. Answer only the question asked: Use only the information provided in the resume. Do not add any extra information or make assumptions.
2. Greetings and other general queries: For non-resume-related questions like greetings or general inquiries, respond appropriately without referring to the resume.
3. Contact details: If asked for contact details, use the following: \n
    - Email: kaushikeebhawsar1999ca@gmail.com \n
4. Frame your answers in such a way that they showcase Kaushikee's importance.
5. No pre-amble and post-amble is required, just answer the question.

Resume:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(), 
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

# st.header("Explore Kaushikee’s Portfolio of Excellence", divider='grey')
st.markdown("""
    <h3 style='text-align: left; padding-top: 35px; border-bottom: 3px solid purple;'>
        Explore Kaushikee’s Portfolio of Excellence
    </h3>""", unsafe_allow_html=True)

side_bar_message = """Hi there, I’m [Kaushikee](https://linkedin.com/in/kaushikee-bhawsar-54127b177), and I’ve developed this virtual assistant, **KBot**, to help you navigate through my professional background efficiently.
Here are some key areas to consider:
1. **Professional Experience**
2. **Technical Skills**
3. **Projects and Achievements**
4. **Education and Certifications**

Feel free to ask anything!

"""

with st.sidebar:
    st.title(':purple_heart: KBot - Kaushikee\'s Virtual Assistant')
    st.markdown(side_bar_message)

initial_message = """
    Hi there!👋 I'm KBot. 
    How can I assist you in exploring Kaushikee's professional expertise and accomplishments? To get started, here are some questions you can ask me:\n
        What are Kaushikee's skills?\n
        How many years of work experience she has?\n
        What are some notable projects she has completed? \n
        Which certifications has she earned?\n
        What makes her a standout candidate?"""
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": 
                                  initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)



# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm checking Kaushikee's profile for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
