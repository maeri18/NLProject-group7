import sys
import os

parent = os.path.dirname(__file__)
mod = os.path.join(parent, "chatbot")
sys.path.append(mod)

from flask import Flask, render_template, request
from chatbot.generate_responses import generate_answer
from collections import deque

# list of previous queries and answers
query_list = deque(maxlen=5)

app = Flask(__name__)

# Main page 
@app.route('/')
def main_page():
    return render_template('main.html')

# chat page
@app.route('/current_chat')
def chat_page(): 
    return render_template('current_chat.html')

# Get the chatbot's response
@app.route('/get_response',methods=["GET", "POST"])
def get_response(): 
    query_text = request.form.get("query") # user's query
    answer = generate_answer(query_text,query_list) # generate answer with gpt and RAG
    query_list.append((query_text, answer.split("<br/>")[0])) # update list of use queries
    return answer



if __name__ == '__main__':
    
    app.run(debug=True, port=9000) #run the app