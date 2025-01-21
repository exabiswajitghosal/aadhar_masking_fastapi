import os
import base64
import requests
from PIL import Image, ImageDraw
import cv2
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
                        "text": "Please extract the 12-digit Aadhaar number from this image. Return ONLY the number with spaces as it appears in the image. nothing else."
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


def find_aadhaar_coordinates(image_path, aadhaar_number):
    """
    Use OpenCV and Tesseract to locate the Aadhaar number in the image.
    """
    try:
        print("I am here ", aadhaar_number)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.medianBlur(gray_image, 3)
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                11, 2)

        # Perform OCR
        data = pytesseract.image_to_data(threshold_image, output_type=pytesseract.Output.DICT)
        print(f"OCR data: {data}")  # Debugging output

        # Normalize text data
        normalized_text = "".join([text for text in data['text'] if text.isdigit()])
        aadhaar_normalized = aadhaar_number.replace(" ", "")

        print(f"Normalized text: {normalized_text}")
        print(f"Normalized Aadhaar number: {aadhaar_normalized}")

        # Check if Aadhaar number exists in normalized text
        if aadhaar_normalized in normalized_text:
            for i, text in enumerate(data['text']):
                # Normalize individual text fragments
                fragment = text.replace(" ", "").replace("'", "").replace("-", "")
                if aadhaar_normalized in fragment:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    print(f"Found Aadhaar number at coordinates: x={x}, y={y}, w={w}, h={h}")
                    return x, y, w, h

        print("Aadhaar number not found with OpenCV.")
        return None
    except Exception as e:
        print(f"Error finding Aadhaar number coordinates: {str(e)}")
        return None


def mask_aadhaar_number(image_path, aadhaar_number, output_path):
    """
    Mask the Aadhaar number in the image using OpenCV coordinates.
    """
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        coordinates = find_aadhaar_coordinates(image_path, aadhaar_number)
        if coordinates is None:
            print("Failed to find Aadhaar number coordinates. Masking aborted.")
            return False

        x, y, w, h = coordinates

        if x is not None:
            # Draw a black rectangle over the Aadhaar number
            draw.rectangle([(x, y), (x + w, y + h)], fill="black")

            # Save the masked image
            image.save(output_path)
            print(f"Successfully saved masked image to: {output_path}")
            return True
        else:
            print("Failed to find Aadhaar number coordinates. Masking aborted.")
            return False

    except Exception as e:
        print(f"Error masking Aadhaar number: {str(e)}")
        return False


def main():

    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return

    input_image_path = "../sample/E01.jpg"
    output_image = "downloads/masked_aadhar.jpg"

    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at path: {input_image_path}")
        return

    print("Starting Aadhaar number extraction...")
    aadhaar_number = extract_aadhaar_with_gpt4(input_image_path)

    if aadhaar_number:
        cleaned_number = aadhaar_number.replace(" ", "")
        print(f"Found Aadhaar number: ********{cleaned_number[-4:]}")

        print("Starting masking process...")
        if mask_aadhaar_number(input_image_path, aadhaar_number, output_image):
            print("Successfully masked Aadhaar number!")
        else:
            print("Failed to mask Aadhaar number.")
    else:
        print("Failed to extract Aadhaar number.")


if __name__ == "__main__":
    main()
