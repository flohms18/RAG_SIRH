from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from ollama import chat
import os, re, html, requests
import pandas as pd
import torch


# ============================================================
# RAG PIPELINE
#   1. LOAD       -> fetch the chosen agreement (IDCC) from Legifrance   (startup + /conv)
#   2. EMBED      -> convert every in-force article into a vector (bge-m3)   (startup + /conv, cached)
#   3. RETRIEVE   -> embed the user request, find the 5 closest articles
#   4. AUGMENT    -> inject those 5 articles + their scores into the LLM prompt
#   5. GENERATE   -> the LLM (gemma via Ollama) answers from those articles only, citing them
# ============================================================


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



### articles -> DataFrame + embeddings (redone each time the user picks a new convention)

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

DEFAULT_IDCC = '1486'  # Syntec

cache = {}                                        # idcc -> (df, embeddings), avoids re-embedding
state = {'idcc': None, 'df': None, 'emb': None}   # convention currently used by /ask


def load_convention(idcc):
    if idcc not in cache:
        # 1. LOAD
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {get_token()}',
        }
        response = requests.post(url_syntec, headers=headers, json={
            'id': idcc  # Numéro de la convention collective (IDCC)
        })
        response.raise_for_status()

        df = pd.DataFrame(extract_articles(response.json().get('sections')))
        if df.empty:
            raise ValueError(f"Aucun article en vigueur trouvé pour l'IDCC {idcc}")
        df = df[df['content'] != ''].reset_index(drop=True)

        # 2. EMBED
        cache[idcc] = (df, model.encode(df['content'].tolist()))

    df, emb = cache[idcc]
    state.update(idcc=idcc, df=df, emb=emb)


load_convention(DEFAULT_IDCC)




### App Url

@app.get('/')
async def index(request : Request):
    return templates.TemplateResponse(request, 'index.html', {
        "idcc" : state['idcc'],
    })


@app.post('/conv')
def form_conv(request : Request, conv_request : str = Form(...)):

    idcc = conv_request.strip()
    error = None

    if not idcc.isdigit():
        error = "Le numéro IDCC doit contenir uniquement des chiffres."
    else:
        try:
            load_convention(idcc)
        except (requests.RequestException, ValueError) as e:
            error = f"Impossible de charger la convention {idcc} : {e}"

    return templates.TemplateResponse(request, 'index.html', {
        "idcc" : state['idcc'],
        "conv_error" : error,
    })









@app.post('/ask')
def ask_form(request : Request, text_request : str = Form(...)):

    idcc, df_conv, emb_conv = state['idcc'], state['df'], state['emb']

    # 3. RETRIEVE
    emb_user = model.encode(text_request)
    compare = model.similarity(emb_user, emb_conv)

    answer = torch.topk(compare, k=min(5, len(df_conv)))

    TOP_5 = answer.indices[0].tolist()
    context = df_conv.iloc[TOP_5].copy()
    context['score'] = [round(x, 2) for x in answer.values[0].tolist()]

    # 4. AUGMENT
    articles = '\n\n'.join(
        f"Article {row.num} (score {row.score}) - {row.section}\n{row.content}"
        for row in context.itertuples()
    )

    SYSTEM = f"""Tu es un assistant RH qui répond uniquement à partir des extraits de la
convention collective fournis ci-dessous.

Règles :
- Réponds uniquement avec les informations présentes dans les extraits.
- Cite le numéro de l'article pour chaque information.
- Si les extraits ne permettent pas de répondre, réponds exactement :
  "Je ne trouve pas cette information dans la convention collective (IDCC {idcc}).
   Je vous invite à contacter votre service RH."
- N'utilise jamais tes connaissances générales.
- Si la réponse dépend du statut ou de la catégorie du salarié (par exemple ouvrier, ETAM, cadre), donne chaque cas.
- Ne réponds jamais à une question sur la situation personnelle d'un salarié tel que le salaire, ne renvoie pas d'articles 
  nommé ou identifiable (son salaire, sa rémunération, ses primes, son
  évaluation, etc.). Dans ce cas, réponds exactement :
  "Je ne peux pas répondre aux questions concernant la situation personnelle, comme le salaire et surtout ne lui donne pas de recommandations
   d'un salarié. Je vous invite à contacter votre service RH."
  Les règles générales de la convention (par exemple les salaires minima
  par position et coefficient) restent autorisées.
- N'utilise pas de texte enrichi (** ou *), c'est extremement important
- N'écris pas de texte en gras (** ou *) et réponds uniquement en français."""

    LLM = f"""Extraits de la convention collective (IDCC {idcc}) :
{articles}

Question :
{text_request}"""

    # 5. GENERATE
    stream = chat(
        model='gemma4:latest',
        messages=[
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': LLM},
        ]
    )

    return templates.TemplateResponse(request, 'index.html', {
        "idcc" : idcc,
        "text_request" : text_request,
        "articles" : context.to_dict(orient='records'),
        "jackpot" : stream['message']['content']
    }
    )












