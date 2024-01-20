from langchain_community.llms import CTransformers
import gradio as gr
import os
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import random

def text_to_pdf(url, filename="output.pdf"):
    video_id = url.split('=')[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    res = ""
    for i in transcript:
        res += " " + i["text"]
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font("DejaVu",'', size=12)
    pdf.multi_cell(0, 10, res)
    pdf.output(filename)
    print(f"{filename} generated!")

local_llm = "llama-2-13b-chat.Q4_K_M.gguf"

config = {
    'max_new_tokens':512,
    'context_length':700,
    'repetition_penalty':1.5,
    'temperature':0.4,
    'top_k':50,
    'top_p':0.9,
    'stream':True,
    'threads':int(os.cpu_count()/2)
}

llm_init = CTransformers(
    model=local_llm,
    model_type="llama",
    lib="avx2",
    **config
)

prompt_template = """Use the following pieces of information to answer the user's question.
Make sure to adhere by the user's request.

Context: {context}
Question: {question}

Relevant Answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def process_transcript():
    loader = PyPDFLoader("output.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # print(texts)
    num = random.randint(0, 10000)
    vectorstore = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space":"cosine"}, persist_directory=f"stores/tcp_cosine_{num}")
    print("Vector Store created!")
    return vectorstore

def get_response(url):
    text_to_pdf(url)
    load_vector_store = process_transcript()
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    retriever = load_vector_store.as_retriever(search_kwargs={"k":1})
    chain_type_kwargs = {"prompt": prompt}
    query = "Summarise the given context in third person perspective in format of bullet points. Make sure to cover the entire content and only provide the crucial important gist in your response. Be as descriptive as you want, but keep the content relevant."
    qa = RetrievalQA.from_chain_type(
        llm=llm_init,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    return response['result']

input = gr.Text(
    label="Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your URL",
    container=False
)

iface = gr.Interface(
    fn=get_response,
    inputs=input,
    outputs="text",
    title="YouTube Video Summarizer",
    description="Enter the URL to the YouTube video that you want to summarize: ",
    allow_flagging=False
)

iface.launch()