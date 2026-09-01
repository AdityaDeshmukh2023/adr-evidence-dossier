from paddleocr import PaddleOCR


def extract_text_from_image(img_path, lang='en'):
    """
    Extracts text from an image using PaddleOCR.
    
    Args:
        img_path (str): Path to the image file.
        lang (str): Language for OCR (default is 'en').
    
    Returns:
        str: Combined text detected from the image.
    """
    # Initialize the PaddleOCR model
    ocr = PaddleOCR(lang=lang, use_angle_cls = True)
    
    # Perform OCR on the image
    result = ocr.ocr(img_path)
    
    # Extract only the detected text
    detected_texts = [line[1][0] for line in result[0]]
    
    # Join the detected texts into a single string
    final_text = " ".join(detected_texts)
    
    return final_text


