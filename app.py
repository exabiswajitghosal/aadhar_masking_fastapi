from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import cv2
import uvicorn
from typing import List
import shutil
from datetime import datetime

# Local Modules
from utils.solution1 import process_aadhar_image
from utils.solution2 import mask_aadhaar

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
        output_filename = f"masked_aadhar_{file.filename}"
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
            
            output_filename = f"masked_aadhar_{file.filename}"
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


@app.post("/mask-aadhar-layoutlm/",
          summary="Mask Aadhaar card using LayoutLM",
          description="Upload an Aadhaar card image to mask sensitive information using LayoutLM")
async def mask_aadhar_layoutlm(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image file
        image_data = await file.read()

        # Process image
        image_path = os.path.join(TEMP_DIR, file.filename)
        output_path = os.path.join(OUTPUT_DIR, f"masked_{file.filename}")
        with open(image_path, "wb") as f:
            f.write(image_data)

        masked_image_path = mask_aadhaar(image_path=image_path, output_path=output_path)

        if not masked_image_path:
            raise HTTPException(status_code=400, detail="No Aadhaar numbers found in the image")

        # Return masked image
        return FileResponse(
            masked_image_path,
            media_type="image/jpeg",
            filename=os.path.basename(masked_image_path)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mask-multiple-aadhars-layoutlm/",
          summary="Mask multiple Aadhaar cards using LayoutLM",
          description="Upload multiple Aadhaar card images to mask sensitive information using LayoutLM")
async def mask_multiple_aadhar_cards_layoutlm(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        masked_images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                continue

            image_data = await file.read()
            image_path = os.path.join(TEMP_DIR, file.filename)
            output_path = os.path.join(OUTPUT_DIR, f"masked_{file.filename}")

            with open(image_path, "wb") as f:
                f.write(image_data)

            masked_image_path = mask_aadhaar(image_path=image_path, output_path=output_path)

            if masked_image_path:
                masked_images.append(masked_image_path)

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