# main.py
from fastapi import FastAPI
from typing import Annotated

app = FastAPI() # This is what will be refrenced in config

@app.get("/fapi")
async def root():
    return {"message": "Hello World"}