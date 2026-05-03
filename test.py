# Create a test.py
from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(api_key="azgxDcCHFQPvx1pbbwdzNY2F1wmxSq95")
print(llm.invoke("Hello, are you awake?"))