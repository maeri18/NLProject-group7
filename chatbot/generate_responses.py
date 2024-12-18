import getpass
import os

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenAI(model="gpt-4o-mini", api_key="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")


prompt_template = hub.pull("rlm/rag-prompt")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(persist_directory="knowledge_base_3", embedding_function=embeddings)

def relevant_document(query_text: str):
    # minimum confidence score to select a document
    minimum_score = 0.3

    # similarity search in the vector store
    results = vector_store.similarity_search_with_score(query=query_text, k=5)
    

    if not results:
        print("No relevant documents found")
        return []

    relevant_documents = []

    # for loop to consider only documents with a high score
    for doc, score in results:
        print(score)
        if score >= minimum_score:
            relevant_documents.append(doc)


    # Return the found answers
    return relevant_documents


def generate_model_gpt_4o_mini(query_text:str,query_list):
  """Function to generate user-friendly answers with gpt-4o-mini"""

  # extract relevant document from knowledge base
  relevant_documents = relevant_document(query_text)

  # convert them into context for gpt
  context = "\n\n".join(doc.page_content for doc in relevant_documents)

  # prompt template
  prompt = (
        f"You are a helpful and friendly assistant answering customer questions. You are having a conversation in which you are helping a user. The user has asked a question (user query), and you have a list of relevant question-answer pairs (relevant answers) to base your answer on. You also have the list of the previous questions that the user asked along with the answers you gave to have more context on how you should answer the user's query.\n\n"
        f"previous user's queries:{query_list}\n\n "
        f"User Query: {query_text}\n\n"
        f"Relevant Answers:\n{context}\n\n"
        f"Your task is to **based on the relevant question-answers and the previous user's queries generate a helpful, user friendly, customer-oriented tone response ** from the lists provided above that directly addresses the user's query. The answer should be clear, concise.\n\n"
        f"Moreover, if the provided relevant answers are unrelated to the user's query, or if there are no relevant answers, say 'I don't know', unless the user's query is a greeting."
        f"Stick to the relevant answers and the previous user queries to answer."
        f"Answer:"
    )
  
  # send prompt to gpt and get response
  response = llm.invoke(prompt)

  return (response.content, relevant_documents)

    
def generate_answer(query_text:str, query_list):
    """Function to return gpt response along with the context documents that were used for generating the response to our GUI application"""
    if query_text.lower() == "exit":
            return "Goodbye!"
    
     
    # Generate a response using the gpt-4o-mini model
    try:
        response, relevant_documents = generate_model_gpt_4o_mini(query_text,query_list)

        # print also the context documents to verify the sources
        context = "**List of context documents:**"+"<br/>*"+"<br/>*".join(doc.page_content for doc in relevant_documents)
        return response +"<br/><br/>" + context
    
    except Exception as e:
        return f"Sorry, an error occurred: {e}"