from typing import Optional
import re
from fastapi import FastAPI

app = FastAPI()

from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords,TFIDF


from river.compose import Pipeline 


pipe_nb = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nb',MultinomialNB()))


with open('raw_chat.txt' , encoding='utf8') as f:
    for line in f:
        line = re.sub('.*- (.*): (.*)', r'\1:\2', line).strip()
        full_message = line.split(':')
        if(len(full_message) == 2):
            pipe_nb.learn_one(full_message[1],full_message[0])
        
@app.get("/predict/{text}")
def predict(text: str):
    return pipe_nb.predict_one(text)

@app.post("/learn/{text}/{label}")
def learn(text: str, label: str):
    pipe_nb.learn_one(text, label)
    return {"status": "ok"}
