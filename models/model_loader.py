# models/model_loader.py

import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from vllm.sampling_params import SamplingParams
from colpali_engine.models import ColPali, ColPaliProcessor
#from vllm import LLM

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from logger import get_logger

logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}

def detect_device():
    """
    Detects the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
print(torch.cuda.is_available())

def load_model(model_choice):
    """
    Loads and caches the specified model.
    """
    global _model_cache

    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    if model_choice == 'qwen':
        device = detect_device()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model.to(device)
        _model_cache[model_choice] = (model, processor, device)
        logger.info("Qwen model loaded and cached.")
        return _model_cache[model_choice]

    # elif model_choice == 'gemini':
    #     # Load Gemini model
    #     api_key = os.getenv("GOOGLE_API_KEY")
    #     if not api_key:
    #         raise ValueError("GOOGLE_API_KEY not found in .env file")
    #     genai.configure(api_key=api_key)
    #     model = genai.GenerativeModel('gemini-1.5-flash-002')  # Use the appropriate model name
    #     return model, None

    # elif model_choice == 'colpali':
    #     # Load ColPali model
    #     device = detect_device()
    #     model_id = "vidore/colpali-v1.2"
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
    #         device_map="auto"
    #     )
    #     processor = ColPaliProcessor.from_pretrained(model_name)
    #     model.to(device)
    #     _model_cache[model_choice] = (model, processor, device)
    #     logger.info("ColPali model loaded and cached.")
    #     return _model_cache[model_choice]
    
    # elif model_choice == "colpali":
    #     device = detect_device()
    #     mistral_models_path = os.path.join(os.getcwd(), 'mistral_models', 'Pixtral')
        
    #     if not os.path.exists(mistral_models_path):
    #         os.makedirs(mistral_models_path, exist_ok=True)
    #         from huggingface_hub import snapshot_download
    #         snapshot_download(repo_id="byaldi/colpali", 
    #                           allow_patterns=["params.json", "consolidated.safetensors", "colpali.json"], 
    #                           local_dir=colpali_models_path)

    #     from mistral_inference.transformer import Transformer
    #     from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    #     from mistral_common.generate import generate

    #     tokenizer = MistralTokenizer.from_file(os.path.join(mistral_models_path, "tekken.json"))
    #     model = Transformer.from_folder(mistral_models_path)
        
    #     _model_cache[model_choice] = (model, tokenizer, generate, device)
    #     logger.info("Pixtral model loaded and cached.")
    #     return _model_cache[model_choice]
    
    elif model_choice == "colpali":
        device = detect_device()
        processor = ColPaliProcessor.from_pretrained(
            #'vidore/colpali-v1.2',
            'vidore/colpali',
            trust_remote_code=True,
            torch_dtype='bfloat16', # bfloat16 instead of auto as specified in the colpali repo
            device_map="cuda:0"  # or "mps" if Apple Silicon #'auto'
        )
        model = ColPali.from_pretrained(
            'vidore/colpali',
            trust_remote_code=True,
            torch_dtype='torch.bfloat16', #'auto',
            device_map='cuda:0' #'auto'
        )
        print("Model loaded:")
        print(model)
    else:
        logger.error(f"Invalid model choice: {model_choice}")
        raise ValueError("Invalid model choice.")
