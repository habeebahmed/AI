import textwrap
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.llms import HuggingFacePipeline


tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")


pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=256,
    device='cuda:0'

)

local_llm = HuggingFacePipeline(pipeline=pipe)

loader = PyPDFLoader('./Test.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})


with open('inst_embeds.pkl','wb') as f:
    pickle.dump(instructor_embeddings,f)

# with open('inst_embeds.pkl','rb') as f:
#     instructor_embeddings = pickle.load(f)

persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=instructor_embeddings,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=instructor_embeddings)


retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)


def process_llm_response(text):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=110) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text



while True:
    try:
        query = input("Enter Question Here: ")
        llm_response = qa_chain(query)
        print(process_llm_response(llm_response['result']))
    except KeyboardInterrupt:
        break