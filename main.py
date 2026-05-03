from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers import EnsembleRetriever
from create_database import get_retrievers
from sentence_transformers import CrossEncoder
from config import PROMPT_V1
from evaluation import run_evaluation


load_dotenv() 


embedding_model = MistralAIEmbeddings(
    model= "mistral-embed"
)


vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


llm = ChatMistralAI(
    model= "mistral-small-2506",

)



# Get retrievers
keyword_retriever, retriever = get_retrievers()



# 3. The Ensemble blends them together. 
# Even if one fails to find a specific word, the other usually catches the meaning.
ensemble_retriever = EnsembleRetriever(
       retrievers=[keyword_retriever, retriever],
       weights=[0.5, 0.5] # Give equal importance to keywords and meaning
)




print("RAG System created")

print("Press 0 to exit")

def rag_pipeline(query):
    docs = ensemble_retriever.invoke(query)
    

    # 1. MOVE THIS CHECK TO THE TOP (Immediately after invoke)
    if not docs:
        return "No relevant documents found.", []

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_docs = sorted(
     zip(docs, scores),
     key=lambda x: x[1],
     reverse=True
)

    # 3. Add a score filter to catch the "garbage" BM25 results
    if ranked_docs[0][1] < 0.2: # If the best match is less than 20% relevant
        return "No relevant documents found.", []
    
    docs = [doc for doc, _ in ranked_docs[:3]]


    if not docs:
        return "No relevant documents found.", []


    else:
    
       context = " "
       sources = []

       for doc in docs:
    
         content = doc.page_content
         page =int(doc.metadata.get("page")) + 1
         chunk_id = doc.metadata.get("chunk_id")

         context += f"{content}\n\n"
         sources.append(f"Page {page}, Chunk {chunk_id}")

    if not any(word.lower() in context.lower() for word in query.split()):
         return "❌ I don't have enough information to answer this."
         
         

    prompt = PROMPT_V1.format(
      context={context},
      query={query}
   )

    response = llm.invoke(prompt)
     
    return response.content, sources



score = run_evaluation(ensemble_retriever, rag_pipeline)

THRESHOLD = 0.7

if score < THRESHOLD:
    raise Exception("❌ Build Failed: Faithfulness too low")
else:
    print("✅ System Passed Evaluation")
      
      

while True:
    query= input("You :")
    if query == "0":
        break

 
    
    answer, sources = rag_pipeline(query)

    # ADD THIS LINE RIGHT HERE:
    print(f"Assistant: {answer}")

    if sources:
     for s in sources:
       print("Source:")
       print("-", s)


      
      
