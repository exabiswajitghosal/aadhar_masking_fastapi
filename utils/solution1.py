from fastapi import HTTPException
import numpy as np
import pytesseract
import re
import easyocr
import cv2
import os
import asyncio


async def process_aadhar_image(image_data: bytes):
    try:
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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

async def process_folder(input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_extensions = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print("No valid image files found in the folder.")
        return

    tasks = []
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        with open(image_path, "rb") as f:
            image_data = f.read()

        processed_image = await process_aadhar_image(image_data)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed and saved: {output_path}")

    print("Processing completed for all Aadhaar cards.")

# Call this function
input_folder_path = "../sample"
output_folder_path = "../temp/sol_1"
os.makedirs(output_folder_path, exist_ok=True)
asyncio.run(process_folder(input_folder_path, output_folder_path))

