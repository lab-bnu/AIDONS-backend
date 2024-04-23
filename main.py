from io import BytesIO
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import zxingcpp
from ultralytics import YOLO


app = FastAPI()

origins = [
	"http://localhost",
	"http://localhost:8080",
	"http://localhost:3000",
	"https://lab-aidons.vercel.app/"
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

def read_image(image_encoded):
	pil_image = Image.open(BytesIO(image_encoded))
	#opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # if opencv
	return pil_image

@app.post("/segment/")
async def segment(file: UploadFile):
	model = YOLO('yolov8n.pt')  # load a pretrained model 
	print("SUCCESSFULLY LOADED=================================")
	results = model('img-test-00.jpg')  # predict on an image
	print("SUCCESSFULLY PREDICTED=================================")
	print(results)
	return {'res': 'done'}



@app.post("/extractinfo/")
async def create_extract_info(file: UploadFile):
	return {"filename": file.filename, "title": "Mon super livre", "author" : "Arthur Le Best", "year" : "2024", "ISBN" : "9782410000757"}


@app.post("/barcode/")
async def read_barcode(file: UploadFile):
	img = read_image(file.file.read())
	results = zxingcpp.read_barcodes(img)
	if len(results) > 0 :
		for result in results:
			return {'code' : f'{result.text}', 'format' : f'{result.format}', 'content' : f'{result.content_type}', 'position' : f'{result.position}'}
	else:
		raise HTTPException(status_code=404, detail="Barcode not found")

@app.get("/")
async def main():
	content = """
<body>
<form action="/barcode/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>

<form action="/segment/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>

</body>
	"""
	return HTMLResponse(content=content)