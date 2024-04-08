# main.py
from fastapi import FastAPI, UploadFile

app = FastAPI() # This is what will be refrenced in config

@app.get("/fapi")
async def root():
    return {"message": "Hello World"}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}

