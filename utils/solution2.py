import pytesseract
from PIL import Image, ImageDraw
import cv2
import re
import os
import glob

# Aadhaar number pattern (4 digits, space, 4 digits, space, 4 digits)
AADHAAR_REGEX = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")


def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


def extract_text_and_bboxes(image_path):
    """Extract Aadhaar numbers and their bounding boxes using Tesseract OCR."""
    preprocessed_image = preprocess_image(image_path)
    ocr_data = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)

    texts, bboxes = [], []
    current_tokens, current_bboxes = [], []

    for i, word in enumerate(ocr_data["text"]):
        if word.strip() and word.isdigit():
            current_tokens.append(word)
            current_bboxes.append([
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["left"][i] + ocr_data["width"][i],
                ocr_data["top"][i] + ocr_data["height"][i]
            ])
        else:
            combined_text = " ".join(current_tokens)
            if AADHAAR_REGEX.fullmatch(combined_text):
                texts.append(combined_text)
                x1 = min(bbox[0] for bbox in current_bboxes[:2])  # First 8 digits
                y1 = min(bbox[1] for bbox in current_bboxes[:2])
                x2 = max(bbox[2] for bbox in current_bboxes[:2])
                y2 = max(bbox[3] for bbox in current_bboxes[:2])
                bboxes.append([x1, y1, x2, y2])
            current_tokens, current_bboxes = [], []

    return texts, bboxes


def mask_aadhaar(image_path, output_path=None):
    """Mask Aadhaar numbers in an image with black rectangles."""
    if not output_path:
        os.makedirs("../temp/sol_2", exist_ok=True)
        output_path = f"../temp/sol_2/masked_{os.path.basename(image_path)}"

    texts, bboxes = extract_text_and_bboxes(image_path)
    if not texts:
        return None

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, fill="black")

    image.save(output_path)
    return output_path


def process_images_in_folder(folder_path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]  # Add other extensions if needed
    image_paths = []

    # Get all image files in the folder
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        print("No images found in the folder.")
        return

    for image_path in image_paths:
        try:
            output_path = mask_aadhaar(image_path)
            if output_path:
                print(f"Masked image saved as: {output_path}")
            else:
                print(f"No Aadhaar numbers found in the image: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    folder_path = "../sample/"
    process_images_in_folder(folder_path)

