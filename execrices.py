## Imports

from fastapi import FastAPI
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pydantic import BaseModel


## Loading data

df = pd.read_csv('exercises.csv')
# print(df.head(10))

documents = []

for _, row in df.iterrows():
    metadata = row.to_dict()
    content = " ".join(str(value) for value in row.values if pd.notnull(value))
    documents.append(Document(page_content=content, metadata=metadata))
    #print(content)
    
## Embeddings and Vector Store
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedder)


## Large Language Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyDGZrDUj0MP6kOtlfe0L6yx9AbYUIp6XpY")

## Memory
memory = ConversationBufferMemory(memory_key="all_chats", return_messages=True)

## Agent Role Prompt

PERSONALITY_PROMPT = """
You are a fitness expert and personal trainer. You have extensive knowledge of exercise science, anatomy, and nutrition. Your goal is to help users achieve their fitness goals by providing personalized exercise recommendations, workout plans, and nutritional advice. You are friendly, supportive, and motivational. You always encourage users to stay consistent with their fitness routines and make healthy lifestyle choices.
Answer only question from the database. If the question is not related to the database, respond by telling the user that you are not able to answer the question.

Chat History: {all_chats}

Context:
{context}

user: {question}

fitness expert:""

"""

prompt_template = PromptTemplate(template =PERSONALITY_PROMPT, input_variables=["all_chats", "question", "context"])

## Conversational Retrieval Chain

chat_assistant = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
)

##FastAPI response transmission



app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    response = chat_assistant.invoke({"question": req.question})
    return {
        "question": req.question,
        "answer": response["answer"],

    }
