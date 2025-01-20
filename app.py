from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import cv2
import numpy as np
import pytesseract
import re
import easyocr
import uvicorn
from typing import List
import shutil
from datetime import datetime
import io

# Setup directories
TEMP_DIR = "temp"
OUTPUT_DIR = "output"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create directories and cleanup old files
    for dir_path in [TEMP_DIR, OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    yield
    # Cleanup on shutdown
    for dir_path in [TEMP_DIR, OUTPUT_DIR]:
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error cleaning up {dir_path}: {e}")

app = FastAPI(
    title="Aadhaar Card Masking API",
    description="API for masking sensitive information in Aadhaar cards",
    lifespan=lifespan
)

async def process_aadhar_image(image_data: bytes):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image)
        except Exception as e:
            # Fallback to pytesseract
            results = []
            try:
                pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
                text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                for i, word in enumerate(text_data["text"]):
                    if int(text_data["conf"][i]) > 60:
                        (x, y, w, h) = (text_data["left"][i], text_data["top"][i], 
                                      text_data["width"][i], text_data["height"][i])
                        results.append([((x, y), (x + w, y + h)), word, text_data["conf"][i]])
            except Exception as e:
                raise HTTPException(status_code=500, 
                                  detail=f"Error during image processing: {str(e)}")

        masked_image = image.copy()
        for result in results:
            if isinstance(result, tuple):
                bbox, text, prob = result
                if re.match(r'\b\d{12}\b', text) or re.match(r'\d{4}\s\d{4}\s\d{4}', text):
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = (int(top_left[0]), int(top_left[1]))
                    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                    text_width = bottom_right[0] - top_left[0]
                    new_bottom_right = (top_left[0] + int(0.66 * text_width), bottom_right[1])
                    cv2.rectangle(masked_image, top_left, new_bottom_right, (0, 0, 0), -1)
            elif isinstance(result, list) and len(result) >= 3:
                bbox, text, prob = result
                if re.match(r'\b\d{12}\b', text) or re.match(r'\d{4}\s\d{4}\s\d{4}', text):
                    top_left = (int(bbox[0][0]), int(bbox[0][1]))
                    bottom_right = (int(bbox[1][0]), int(bbox[1][1]))
                    text_width = bottom_right[0] - top_left[0]
                    new_bottom_right = (top_left[0] + int(0.66 * text_width), bottom_right[1])
                    cv2.rectangle(masked_image, top_left, new_bottom_right, (0, 0, 0), -1)

        return masked_image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/mask-aadhar/", 
          summary="Mask Aadhaar card",
          description="Upload an Aadhaar card image to mask sensitive information")
async def mask_aadhar_card(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Process image
        masked_image = await process_aadhar_image(image_data)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"masked_aadhar_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save masked image
        cv2.imwrite(output_path, masked_image)
        
        # Return masked image
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename=output_filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mask-multiple-aadhars/",
          summary="Mask multiple Aadhaar cards",
          description="Upload multiple Aadhaar card images to mask sensitive information")
async def mask_multiple_aadhar_cards(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        masked_images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                continue
                
            image_data = await file.read()
            masked_image = await process_aadhar_image(image_data)
            
            output_filename = f"masked_aadhar_{timestamp}_{idx}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            cv2.imwrite(output_path, masked_image)
            masked_images.append(output_path)
        
        if not masked_images:
            raise HTTPException(status_code=400, detail="No valid images processed")
            
        # Create ZIP file containing all masked images
        zip_filename = f"masked_aadhars_{timestamp}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)
        
        # Create a temporary directory for ZIP creation
        temp_zip_dir = os.path.join(TEMP_DIR, f"zip_{timestamp}")
        os.makedirs(temp_zip_dir, exist_ok=True)
        
        # Copy masked images to temporary directory
        for img_path in masked_images:
            shutil.copy2(img_path, temp_zip_dir)
        
        # Create ZIP file
        shutil.make_archive(zip_path[:-4], 'zip', temp_zip_dir)
        
        # Cleanup temporary directory
        shutil.rmtree(temp_zip_dir)
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)