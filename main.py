# main.py
from fastapi import FastAPI

app = FastAPI() # This is what will be refrenced in config

@app.get("/fapi")
async def root():
    return {"message": "Hello World"}