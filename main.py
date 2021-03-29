from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette import status
import shutil
import os

from predict import predict_faceshape


UPLOAD_PATH = "./"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_image(request: Request, image: UploadFile = File(...)):
    image_path = os.path.join(UPLOAD_PATH, image.filename)
    with open(image_path, "wb") as buffers:
        shutil.copyfileobj(image.file, buffers)
    data = predict_faceshape(image_path)
    os.remove(image_path)
    return templates.TemplateResponse("index.html", {"request": request, "result": data})
