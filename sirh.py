from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer


### App Config 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

### App Url 

@app.get('/')
async def index(request : Request)
    return templates.TemplateResponse(request, 'index.html', {

    })





