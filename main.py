from io import BytesIO
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import traceback
import cv2, zxingcpp
import numpy as np




app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://lab-aidons.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencvImage


@app.post("/extractinfo/")
async def create_extract_info(file: UploadFile):
    return {"filename": file.filename, "title": "Mon super livre", "author" : "Arthur Le Best", "year" : "2024", "ISBN" : "9782410000757"}


@app.post("/barcode/")
async def read_barcode(file: UploadFile):
    img = read_image(file.file.read())
    results = zxingcpp.read_barcodes(img)
    for result in results:
        return {'ISBN' : result.text}
    if len(results) == 0:
	    raise HTTPException(status_code=404, detail="Barcode not found")
'''   print('======================', len(results))
    if len(results)>0:
        return {"data": results[0].text,  "type": results[0].format, "rect": results[0].position, "quality": results[0].content_type}
    else:
	    raise HTTPException(status_code=404, detail="Barcode not found")
  img = read_image(file.file.read()) # PIL Image
    from pyzbar.pyzbar import decode
    decoded_list = decode(img)
    print(decoded_list)
    if len(decoded_list) > 0: 
        return {"data": decoded_list[0].data,  "type": decoded_list[0].type, "rect": decoded_list[0].rect, "quality": decoded_list[0].quality, "orient": decoded_list[0].orientation}
    else:
        raise HTTPException(status_code=404, detail="Barcode not found")'''
    


@app.get("/")
async def main():
    content = """
<body>
<form action="/barcode/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)