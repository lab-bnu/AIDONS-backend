from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.post("/extractinfo/")
async def create_extract_info(file: UploadFile):
    return {"filename": file.filename, "title": "Mon super livre", "author" : "Arthur Le Best", "year" : "2024", "ISBN" : "9782410000757"}



@app.get("/")
async def main():
    content = """
<body>
<form action="/extractinfo/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)