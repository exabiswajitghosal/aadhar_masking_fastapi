import os
import base64
import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import pytesseract

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def encode_image(image_path):
    """
    Encode image to base64 string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at path: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None


def extract_aadhaar_with_gpt4(image_path):
    """
    Extract Aadhaar number using GPT-4 Vision API
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract the 12-digit number from this image. Return ONLY the number with spaces as it appears in the image. nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            print(f"API Error Response: {response.text}")
            return None

        response_data = response.json()
        aadhaar_number = response_data['choices'][0]['message']['content'].strip()

        # Remove all spaces and check if it's a 12-digit number
        cleaned_number = aadhaar_number.replace(" ", "")
        if len(cleaned_number) == 12 and cleaned_number.isdigit():
            return aadhaar_number  # Return original number with spaces
        else:
            print(f"Invalid Aadhaar number format received: {aadhaar_number}")
            return None

    except Exception as e:
        print(f"Error extracting Aadhaar number: {str(e)}")
        return None



def mask_aadhaar_number(image_path, aadhaar_number, output_path):
    aadhaar_number_digits = aadhaar_number.split(" ")[0:2]

    image = Image.open(image_path)

    # Extract data from the image using pytesseract
    data = pytesseract.image_to_data(image, lang='eng+hin', output_type=pytesseract.Output.DICT)

    # Locate the Aadhaar number's bounding box
    aadhaar_coordinates = None

    image_with_mask = image.copy()

    for i, text in enumerate(data['text']):
        for digit in range(len(aadhaar_number_digits) * 2):
            if text.strip() in aadhaar_number_digits:
                # Get bounding box coordinates
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                aadhaar_coordinates = (x, y, x + w, y + h)
                draw = ImageDraw.Draw(image_with_mask)
                draw.rectangle(aadhaar_coordinates, fill="black")
            break

    # If Aadhaar number is found, mask it
    if aadhaar_coordinates:
        image_with_mask.save(output_path)

        print(f"Aadhaar number masked successfully! Saved to {output_path}")
        return True
    else:
        print("Aadhaar number not found in the image.")
        return False
