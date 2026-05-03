
from dotenv import load_dotenv

load_dotenv()
# evaluation.py

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# 1. Evaluator LLM (FREE MODEL)
# -------------------------------
hf_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",   # you can change to flan-t5-small if slow
    max_new_tokens=50,
    temperature=0.7
)

evaluator_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -------------------------------
# 2. Golden Dataset
# -------------------------------
golden_data = [
    {
        "question": "What is the primary goal of the book 'Fundamentals of Deep Learning'?",
        "answer": "The book aims to bridge the gap between jargon-filled research papers and scattered tutorials by providing intuition, mathematical foundations, and practical PyTorch implementations for deep learning[cite: 36, 100]."
    },
    {
        "question": "What are the three core fields of mathematics that form the foundation of deep learning according to Chapter 1?",
        "answer": "Deep learning is a culmination of achievements in calculus, linear algebra, and probability[cite: 105]."
    }
     
]

# -------------------------------
# 3. Faithfulness Check Function
# -------------------------------
def check_faithfulness(question, generated_answer, context):
    prompt = f"""
You are a strict judge.

Question: {question}

Answer: {generated_answer}

Context:
{context}

Task:
Check if the answer is supported by the context.

Rules:
- If answer contains extra info → NO
- If supported → YES

Only reply YES or NO.
"""

    response = evaluator_llm.invoke(prompt)
    output = response.strip().upper()

    if "YES" in output:
        return "YES"
    else:
        return "NO"


# -------------------------------
# 4. Main Evaluation Function
def run_evaluation(retriever, rag_pipeline):
    total = len(golden_data)
    correct = 0

    for item in golden_data:
        question = item["question"]

        print("\n------------------------")
        print(f"Question: {question}")

        # Step 1: Retrieve docs
        docs = retriever.invoke(question)
        docs = docs[:3]

        context = ""
        for doc in docs:
            context += doc.page_content + "\n"

        # Step 2: Generate answer
        generated_answer = rag_pipeline(question)

        print(f"Generated Answer: {generated_answer}")

        # Step 3: Faithfulness check
        result = check_faithfulness(question, generated_answer, context)

        print(f"Faithfulness: {result}")

        if result == "YES":
            correct += 1

    score = correct / total
    print(f"\nFinal Score: {score}")

    return score

