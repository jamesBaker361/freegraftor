import os
import hashlib
from safetensors.torch import load_file, save_file
from pathlib import Path
from PIL import Image
from torchvision import transforms

def get_cache_path(image_path: str, cache_dir: str= './image_info_cache', concept_name: str=None) -> str:
    sha = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
    file_path = f"{concept_name}_{sha}.safetensors" if concept_name else f"{sha}.safetensors"
    return os.path.join(cache_dir, file_path)

def load_image_info(image_path: str, cache_dir: str= './image_info_cache', concept_name: str = None):
    cache_path = get_cache_path(image_path, cache_dir, concept_name)
    if not Path(cache_path).is_file():
        return None
    
    image_info = load_file(cache_path)
    return image_info

def save_image_info(image_info, image_path: str, cache_dir: str= './image_info_cache', concept_name: str = None):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    save_file(image_info, get_cache_path(image_path, cache_dir, concept_name))

def save_image(pil_image, cache_dir='./image_cache', suffix=""):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    sha = hashlib.sha256(pil_image.tobytes()).hexdigest()
    image_path = os.path.join(cache_dir, f'{sha}.png')
    if suffix:
        image_path = image_path.replace('.png', f'_{suffix}.png')
    pil_image.save(image_path)
    return image_path

def load_image(image_path=None, pil_image=None, to_tensor=False):
    if image_path is not None:
        image_pil = Image.open(image_path).convert("RGB")
    elif pil_image is not None:
        image_pil = pil_image
    else:
        raise NotImplementedError
    
    if to_tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(image_pil)
        return image_pil, image_tensor  
    else:
        return image_pil