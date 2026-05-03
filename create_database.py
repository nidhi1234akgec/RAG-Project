#load pdf
#split into chunks
#create the embeddings
#store into chroma 


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv


load_dotenv()

def get_retrievers():

# Load the documents
  data = PyPDFLoader("Document Loaders/deep learning.pdf")
  docs = data.load()

 # Split into chunks
  splitter = RecursiveCharacterTextSplitter(
      chunk_size = 800,
      chunk_overlap= 100
  )

  chunks = splitter.split_documents(docs)


  for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i 


  embedding_model = MistralAIEmbeddings(
    model = "mistral-embed"
)
 # Create vectorstore
  vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "chroma_db"
)

  keyword_retriever = BM25Retriever.from_documents(chunks)
  # Change your retriever configuration to this:
  retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.8  # Adjust this between 0 and 1
    }
)

  return keyword_retriever, retriever


