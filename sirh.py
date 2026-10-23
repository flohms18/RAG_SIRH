from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os, requests


### App Config 

app = FastAPI()
templates = Jinja2Templates(directory="templates")
load_dotenv()


### App Url 

@app.get('/')
async def index(request : Request):
    return templates.TemplateResponse(request, 'index.html', {

    })

### API CONFIG

url = os.getenv('API_URL')
id = os.getenv('OATH_ID')
secret = os.getenv('OATH_SECRET')
token = os.getenv('TOKEN_URL')

def get_token():
    response = requests.post(
        token,
        data={                                  
            'grant_type': 'client_credentials',
            'client_id': id,
            'client_secret': secret,
            'scope': 'openid',
        },
    )
    response.raise_for_status()                 
    return response.json()['access_token']


def jack():
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {get_token()}',
    }
    legi_data = requests.post(url, headers=headers, json={
        'id' : '1486'
    })
    print(legi_data)
    print(legi_data.text)

jack()

### App RAG through Post Request












