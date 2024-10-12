from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer , InputExample , util
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import numpy as np
import torch
import json
import requests
from neo4j import GraphDatabase, basic_auth
import faiss
import pandas as pd


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "1234567890")
driver = GraphDatabase.driver(URI, auth=AUTH)


def run_query(query, parameters=None):
   with GraphDatabase.driver(URI, auth=AUTH) as driver:
       driver.verify_connectivity()
       with driver.session() as session:
           result = session.run(query, parameters)
           return [record for record in result]
   driver.close()

cypher_query_greeting = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_asking = '''
MATCH (n:Asking) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_nonsense = '''
MATCH (n:Nonsense) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_laughing = '''
MATCH (n:Laughing) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_end_conversation = '''
MATCH (n:end_conversation) RETURN n.name as name, n.msg_reply as reply;
'''
cypher_query_question = '''
MATCH (n:Question) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
asking_corpus = []
nonsense_corpus = []
laughing_corpus = []
question_corpus = []
end_conversation_corpus = []
greeting_vec = None
asking_vec = None
nonsense_vec = None
laughing_vec = None
ending_vec = None
question_vec = None


results1 = run_query(cypher_query_greeting)
for record in results1:
    greeting_corpus.append(record['name'])
greeting_corpus = list(set(greeting_corpus))
greeting_vec = model.encode(greeting_corpus)

results2 = run_query(cypher_query_asking)
for record in results2:
    asking_corpus.append(record['name'])
asking_corpus = list(set(asking_corpus)) 
asking_vec = model.encode(asking_corpus)

results3 = run_query(cypher_query_laughing)
for record in results3:
    laughing_corpus.append(record['name'])
laughing_corpus = list(set(laughing_corpus))
laughing_vec = model.encode(laughing_corpus)

results4 = run_query(cypher_query_nonsense)
for record in results4:
    nonsense_corpus.append(record['name'])
nonsense_corpus = list(set(nonsense_corpus))
nonsense_vec = model.encode(nonsense_corpus)

results5 = run_query(cypher_query_end_conversation)
for record in results5:
    end_conversation_corpus.append(record['name'])
end_conversation_corpus = list(set(end_conversation_corpus))
ending_vec = model.encode(end_conversation_corpus)

results6 = run_query(cypher_query_question)
for record in results6:
    question_corpus.append(record['name'])
question_corpus = list(set(question_corpus))
question_vec = model.encode(question_corpus)

session = requests.Session()
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Adjust URL if necessary
headers = {
    "Content-Type": "application/json"
}

def ollama(input, reply , score ,sent , fast_check):
    global session , OLLAMA_API_URL,headers


    if score < 0.5 :
        suffix = '\n\n- เรียบเรียงการตอบกลับจาก Neo4j ด้วย Ollama \U0001F999 -'
        payload = {
            "model": "supachai/llama-3-typhoon-v1.5", 
            "prompt": f"""ข้อความ : {input} ?, การตอบกลับ : {reply} , ช่วยสร้างคำตอบกลับของข้อความนี้ใหม่ โดยมีใจความเหมือนเดิมเป็นประโยคที่เข้าใจง่าย ไม่เกิน 30 คำ ตอบเฉพาะส่วนที่เป็นคำตอบเท่านั้น โดยใช้โทนคำพูดแบบเด็กผู้ชายที่สุภาพ น่ารักๆ พร้อมแสดงความยินดีทุกครั้งที่ได้ตอบคำถาม""",
            "stream": False
        }
 
    else : 
        suffix = '\n\n- สร้างข้อความตอบกลับด้วย Ollama \U0001F999 -'
        payload = { 
            "model": "supachai/llama-3-typhoon-v1.5",  
            "prompt": f"""ข้อความ : {sent} ? ,ช่วยตอบกลับข้อความคำถามด้วยประโยคที่มีเหตุผล ในบริบทของการขับขี่ กระชับ ไม่เกิน 30 คำ ตอบเฉพาะส่วนที่เป็นคำตอบเท่านั้น โดยใช้โทนคำพูดแบบเด็กผู้ชายที่สุภาพ น่ารักๆ พร้อมแสดงความยินดีทุกครั้งที่ได้ตอบคำถาม""",
            "stream": False
        }  
    response = session.post(OLLAMA_API_URL, headers=headers, data= json.dumps(payload))

    if response.status_code == 200:
        response_data = response.text 
        data = json.loads(response_data) 
        decoded_text = data["response"] 
        return  decoded_text + suffix
       
    else:
        return (f"Failed to get a response: {response.status_code}, {response.text}")
    

def faiss_index(vector):
    vector_dimension = vector.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vector)
    index.add(vector)
    return index

def compute_nearest(vector , sentence , corpus):
    df = pd.DataFrame(corpus, columns=['contents'])
    index = faiss_index(vector)
    search_vector = model.encode(sentence)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    k = index.ntotal
    distances , ann = index.search(_vector , k = k)
    results = pd.DataFrame({'distances':distances[0],  'ann' : ann[0]})
    merge = pd.merge(results, df ,left_on = 'ann' , right_index = True)
    return  merge['contents'][0] , merge['distances'][0] ,


def neo4j_search(neo_query):
   results = run_query(neo_query)
   for record in results:
       response_msg = record['reply']
   return response_msg     

def compute_response(sentence):


   greeting_word , greeting_score = compute_nearest(greeting_vec , sentence , greeting_corpus)
   asking_word , asking_score = compute_nearest(asking_vec , sentence , asking_corpus)
   nonsense_word , nonsense_score = compute_nearest(nonsense_vec , sentence , nonsense_corpus)
   laughing_word , laughing_score = compute_nearest(laughing_vec ,sentence , laughing_corpus)
   ending_word , ending_score = compute_nearest(ending_vec , sentence , end_conversation_corpus)
   question_word , question_score = compute_nearest(question_vec , sentence , question_corpus)

   print(f'Distance Greeting [{greeting_word}]: {greeting_score}')
   print(f'Distance Asking [{asking_word}]: {asking_score}')
   print(f'Distance Nonsense [{nonsense_word}]: {nonsense_score}')
   print(f'Distance Laughin [{laughing_word}]: {laughing_score}')
   print(f'Distance Ending [{ending_word}]: {ending_score}')
   print(f'Distance Question [{question_word}]: {question_score}')
   

   min_dis = [greeting_score , asking_score , nonsense_score , laughing_score , ending_score , question_score]
   matching  = [greeting_word , asking_word , nonsense_word , laughing_word , ending_word , question_word]
   min_index = min_dis.index(min(min_dis))
   cypher_match = ['Greeting' , 'Asking' , 'Nonsense' , 'Laughing' , 'end_conversation' , 'Question']
   print(cypher_match[min_index]) 
   print(matching[min_index])
   My_cypher = f"""MATCH (n:{cypher_match[min_index]}) where n.name ="{matching[min_index]}" RETURN n.msg_reply as reply"""
   my_msg  = neo4j_search(My_cypher)
   print(my_msg) 
   print(min(min_dis))
   print(sentence)
   gen_msg = ollama(matching[min_index] , my_msg , min(min_dis) , sentence , cypher_match[min_index])
   return gen_msg


   

app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)                    
    try:
        json_data = json.loads(body)                         
        access_token = 'oBE0/hWRQqMVkNZXXbTKhBeBVA3IDjsE7WOA3SXtkmcwQRT+/nigb2xzoVp5InwWGljuCRw78QMTvOKHrQb16Ov6dyAnxaTDbWBUhM7pe1QzPIFFtXnKUvv9vklnJ0Lw2Scid7OfQ9M1vRIDjPFQ0wdB04t89/1O/w1cDnyilFU=' 
        secret = '33ffe8ae4acd6532eb2acd4857bb3a67'
        line_bot_api = LineBotApi(access_token)              
        handler = WebhookHandler(secret)                     
        signature = request.headers['X-Line-Signature']      
        handler.handle(body, signature)                      
        msg = json_data['events'][0]['message']['text']      
        tk = json_data['events'][0]['replyToken']
        response_msg = compute_response(msg)       
        line_bot_api.reply_message( tk, TextSendMessage(text=response_msg) ) 
        print(msg, tk) 
    except:
        print(body)         

    return 'OK'                 
if __name__ == '__main__':
    app.run(port=5000 , debug= True)




@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)                    
    try:
        json_data = json.loads(body)                         
        access_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 
        secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'
        line_bot_api = LineBotApi(access_token)              
        handler = WebhookHandler(secret)                     
        signature = request.headers['X-Line-Signature']      
        handler.handle(body, signature)                      
        msg = json_data['events'][0]['message']['text']      
        tk = json_data['events'][0]['replyToken']
        response_msg = compute_response(msg)       
        line_bot_api.reply_message( tk, TextSendMessage(text=response_msg) ) 
        print(msg, tk) 
    except:
        print(body)         
                                         
    return 'OK'                 



if __name__ == '__main__':
    app.run(port=5000 , debug= True)



