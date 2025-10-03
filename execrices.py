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

df = pd.read_csv("exercises.csv").fillna("")
# print(df.head(10))



documents = []

instruction_cols = [col for col in df.columns if col.startswith("instructions/")]
secondary_muscles_cols = [col for col in df.columns if col.startswith("secondaryMuscles/")]

documents = []

for _, row in df.iterrows():
    instructions = [str(row[col]) for col in instruction_cols if pd.notna(row[col])]
    secondary_muscles = [str(row[col]) for col in secondary_muscles_cols if pd.notna(row[col])]
    
    content = f"""
            Exercise: {row['exercise_name']}
            Muscle Group: {row['muscle_group']}
            Equipment: {row['equipment']}
            Secondary Muscles: {', '.join(secondary_muscles) if secondary_muscles else 'None'}
            Instructions: {' '.join(instructions) if instructions else 'No instructions available'}
            """
    documents.append(Document(page_content=content.strip(), metadata=row.to_dict()))



    #print(content)
    
## Embeddings and Vector Store
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedder)


## Large Language Model
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", api_key="AIzaSyDn3x1eLz0IPUwJmTuJ0kjDEpk7q_YQ9j4")

## Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

## Agent Role Prompt

PERSONALITY_PROMPT = """
You are a fitness expert and personal trainer. You have extensive knowledge of exercise science, anatomy, and nutrition. Your goal is to help users achieve their fitness goals by providing personalized exercise recommendations, workout plans, and nutritional advice. You are friendly, supportive, and motivational. You always encourage users to stay consistent with their fitness routines and make healthy lifestyle choices.
Answer only question from the database. If the question is not related to the database, respond by telling the user that you are not able to answer the question.

Chat History: {chat_history}

Context:
{context}

user: {question}

fitness expert:

"""

prompt_template = PromptTemplate(template =PERSONALITY_PROMPT, input_variables=["chat_history", "question", "context"])

## Conversational Retrieval Chain

chat_assistant = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
)

##FastAPI response transmission


##FastAPI response transmission

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = chat_assistant.invoke({"question": req.question})
        print("Raw response:", response)   # so you can see what keys exist
        return {
            "question": req.question,
            "answer": response.get("answer") or response.get("result") or response
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


