from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from ollama import chat
import os, re, html, requests
import pandas as pd
import torch


### App Config 

app = FastAPI()
templates = Jinja2Templates(directory="templates")
load_dotenv()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = SentenceTransformer('BAAI/bge-m3')

### API CONFIG

url_syntec = os.getenv('API_URL_SYNTEC')
url_code = os.getenv('API_URL_CODE')
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



headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {get_token()}',
}
syntec = requests.post(url_syntec, headers=headers, json={
    'id' : '1486'
})
syntec.raise_for_status()
syntec_data = syntec.json()


### Syntec articles -> DataFrame + embeddings (done once at startup)

IN_FORCE = {'VIGUEUR', 'VIGUEUR_ETEN', 'VIGUEUR_NON_ETEN'}

def clean_html(text):
    text = re.sub(r'<br\s*/?>|</p>', '\n', text or '')
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\n\s*\n+', '\n', html.unescape(text)).strip()

def extract_articles(sections, path=()):
    for section in sections or []:
        section_path = path + (section.get('title') or '',)
        for article in section.get('articles') or []:
            if article.get('etat') in IN_FORCE:
                yield {
                    'id': article['id'],
                    'num': article.get('num'),
                    'section': ' > '.join(t for t in section_path if t),
                    'content': clean_html(article.get('content')),
                }
        yield from extract_articles(section.get('sections'), section_path)

df_syntec = pd.DataFrame(extract_articles(syntec_data['sections']))
df_syntec = df_syntec[df_syntec['content'] != ''].reset_index(drop=True)

emb_syntec = model.encode(df_syntec['content'].tolist())






### App Url 

@app.get('/')
async def index(request : Request):
    return templates.TemplateResponse(request, 'index.html', {

    })



@app.post('/ask')
def ask_form(request : Request, text_request : str = Form(...)):

    emb_user = model.encode(text_request)
    compare = model.similarity(emb_user, emb_syntec)

    answer = torch.topk(compare, k=min(5, len(df_syntec)))

    TOP_5 = answer.indices[0].tolist()
    context = df_syntec.iloc[TOP_5].copy()
    context['score'] = [round(x, 2) for x in answer.values[0].tolist()]

    articles = '\n\n'.join(
        f"Article {row.num} (score {row.score}) - {row.section}\n{row.content}"
        for row in context.itertuples()
    )

    LLM = f"""
            Pick the 3 articles that match the most with the user request.
            Use only the articles listed below and explain briefly why each one matches.
            Give me the similarity score for each after the article name.
            Don't write bold text (** or *) and answer in French language and French only

            Articles:
            {articles}

            User request:
            {text_request}
            """

    stream = chat(
        model='gemma4:latest',
        messages=[{'role': 'user', 'content': LLM}]
    )

    return templates.TemplateResponse(request, 'index.html', {
        "text_request" : text_request,
        "articles" : context.to_dict(orient='records'),
        "jackpot" : stream['message']['content']
    }
    )




### App RAG through Post Request












