# models/responder.py

from models.model_loader import load_model
from transformers import GenerationConfig
from dotenv import load_dotenv
from logger import get_logger
from openai import OpenAI
from PIL import Image
import torch
import base64
import os
import io


logger = get_logger(__name__)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(images, query, session_id, resized_height=280, resized_width=280, model_choice='gpt4'):
    """
    Generates a response using the selected model based on the query and images.
    """
    try:
        logger.info(f"Generating response using model '{model_choice}'.")
        
        # Convert resized_height and resized_width to integers
        resized_height = int(resized_height)
        resized_width = int(resized_width)
        
        # Ensure images are full paths
        full_image_paths = [os.path.join('static', img) if not img.startswith('static') else img for img in images]
        
        # Check if any valid images exist
        valid_images = [img for img in full_image_paths if os.path.exists(img)]
        
        if not valid_images:
            logger.warning("No valid images found for analysis.")
            return "No images could be loaded for analysis."
        
        if model_choice == 'gpt4':
        #elif model_choice == 'gpt4':
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            
            try:
                query += ". Answer the question using the provided documents. Be sure to include the relevant section names and numbers, page numbers, and any parenthetical references found in the text that you cite from the document. If the answer is not found in these page simply say that you cannot find the answer in the documents provided."

                content = [{"type": "text", "text": query}]
                print(query)
                print(query)
                for img_path in valid_images:
                    logger.info(f"Processing image: {img_path}")
                    if os.path.exists(img_path):
                        base64_image = encode_image(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    else:
                        logger.warning(f"Image file not found: {img_path}")
                
                if len(content) == 1:  # Only text, no images
                    return "No images could be loaded for analysis."
          
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=8000
                )
                
                generated_text = response.choices[0].message.content
                logger.info("Response generated using GPT-4 model.")
                return generated_text
            
            except Exception as e:
                logger.error(f"Error in GPT-4 processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing the images: {str(e)}"
        else:
            logger.error(f"Invalid model choice: {model_choice}")
            return "Invalid model selected."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred while generating the response: {str(e)}"
