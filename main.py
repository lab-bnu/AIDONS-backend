from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from PIL import Image
import zxingcpp
from ultralytics import YOLO
import pandas as pd
import os
import json
from loguru import logger
import sys
import pytesseract
import configparser

app = FastAPI()

model = YOLO('best.pt')  # load a pretrained model 

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

#====================================================================================================================
def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(binary_image).convert("RGB")
#	    input_image = Image.open(BytesIO(binary_image)).convert("RGB")

    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


def crop_image_by_predict(image: Image, predict: pd.DataFrame, crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)



################################# Models #####################################


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict


####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)
    
    print(predict)

    # Step 4: Select detect obj return info
    # here you can choose what data to send to the result
    
    # If there are ISBN detected : try the barcode reader, and if not, try OCR. Once one of them is found, then return
    # Else, return OCRed title, name, and editor
    if len(predict[(predict['name'] == 'ISBN')]) > 0:
        cropped_image = crop_image_by_predict(input_image, predict, 'ISBN')
        isbn = zxingcpp.read_barcodes(cropped_image)
        print(isbn)
        if len(isbn) > 0 :
	        for result in isbn:
                 return {'code' : f'{result.text}', 'format' : f'{result.format}', 'content' : f'{result.content_type}', 'position' : f'{result.position}'}
        else:
            raise HTTPException(status_code=404, detail="Barcode not found")
    
    objects = predict['name'].values

    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(predict.to_json(orient='records'))
    
    for item in result['detect_objects']:
        if item['name'] != 'ISBN':
            print(item)
            #config = configparser.ConfigParser()
            #config.readfp(open(r'./config.txt'))
            #print(config.get('TESSERACT', 'PATH'))
            pytesseract.pytesseract.tesseract_cmd = '<path_to_tesseract.exe>'   #config.get('TESSERACT', 'PATH')
            txt = pytesseract.image_to_string(crop_image_by_predict(input_image, predict, item['name'])) #for now only crop the first in the category, TODO change that 
            item = item.update({'Text': txt})
    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result


def try_barcode_from_file(file: UploadFile):
	img = get_image_from_bytes(file.file.read())
	results = zxingcpp.read_barcodes(img)
	if len(results) > 0 :
		for result in results:
			return {'code' : f'{result.text}', 'format' : f'{result.format}', 'content' : f'{result.content_type}', 'position' : f'{result.position}'}
	else:
		return False

#====================================================================================================================


@app.post("/segmentation_ocr_dummy/")
async def segmentation_ocr_dummy(file: UploadFile):
    """
    Test - Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections, always the same for testing purposes.
    """
    input_image = get_image_from_bytes(file)
    return {"detect_objects":[{"xmin":172.3708953857,"ymin":275.1319580078,"xmax":774.1133422852,"ymax":491.0144042969,"confidence":0.9650346637,
                            "class":3,"name":"Titre","Text":"LA GRANDE\nBEU VERIE\n"},
                           {"xmin":319.3504943848,"ymin":126.0726394653,"xmax":638.5608520508,"ymax":187.2577972412,"confidence":0.7958808541,
                            "class":0,"name":"Auteur","Text":"RENE DAUMAL\n"},
                           {"xmin":219.3504943848,"ymin":26.0726394653,"xmax":538.5608520508,"ymax":87.2577972412,"confidence":0.60,
                            "class":1,"name":"ISBN","Text":"123456789\n"}],
         "detect_objects_names":"Titre, Auteur, ISBN"}


@app.get("/")
async def main():
	content = """
<form action="/img_object_detection_to_json/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>

</body>
	"""
	return HTMLResponse(content=content)
