import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any, Optional, Union
import cv2
import base64
from io import BytesIO
import torch
import torchvision.transforms as transforms

def load_image(image_path: Union[str, BytesIO, Image.Image]) -> Image.Image:
    """Load image from various sources"""
    if isinstance(image_path, str):
        return Image.open(image_path)
    elif isinstance(image_path, BytesIO):
        return Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        return image_path
    else:
        raise ValueError(f"Unsupported image type: {type(image_path)}")

def image_to_base64(image: Union[Image.Image, np.ndarray]) -> str:
    """Convert image to base64 string"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def resize_image(
    image: Image.Image,
    max_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """Resize image with optional aspect ratio preservation"""
    if maintain_aspect_ratio:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)

def draw_bounding_box(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    label: Optional[str] = None,
    color: str = "red",
    width: int = 2
) -> Image.Image:
    """Draw bounding box on image"""
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Draw rectangle
    draw.rectangle(bbox, outline=color, width=width)
    
    # Draw label if provided
    if label:
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        # Get text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background for text
        label_bbox = (bbox[0], bbox[1] - text_height - 4, 
                     bbox[0] + text_width + 4, bbox[1])
        draw.rectangle(label_bbox, fill=color)
        
        # Draw text
        draw.text((bbox[0] + 2, bbox[1] - text_height - 2), 
                 label, fill="white", font=font)
        
    return image_copy

def extract_region(
    image: Image.Image,
    bbox: Tuple[int, int, int, int]
) -> Image.Image:
    """Extract a region from image"""
    return image.crop(bbox)

def create_image_grid(
    images: List[Image.Image],
    grid_size: Optional[Tuple[int, int]] = None,
    padding: int = 10
) -> Image.Image:
    """Create a grid of images"""
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
        
    # Ensure all images have the same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Create canvas
    canvas_width = cols * max_width + (cols + 1) * padding
    canvas_height = rows * max_height + (rows + 1) * padding
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * (max_width + padding) + padding
        y = row * (max_height + padding) + padding
        
        # Center image if smaller than max size
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        
        canvas.paste(img, (x + x_offset, y + y_offset))
        
    return canvas

def compare_images(
    image1: Image.Image,
    image2: Image.Image,
    method: str = "mse"
) -> float:
    """Compare two images using various metrics"""
    # Convert to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Ensure same size
    if arr1.shape != arr2.shape:
        # Resize to match
        image2_resized = image2.resize(image1.size)
        arr2 = np.array(image2_resized)
        
    if method == "mse":
        # Mean Squared Error
        return np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
    elif method == "ssim":
        # Structural Similarity Index
        from skimage.metrics import structural_similarity
        
        # Convert to grayscale if needed
        if len(arr1.shape) == 3:
            gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = arr1, arr2
            
        return structural_similarity(gray1, gray2)
    else:
        raise ValueError(f"Unknown comparison method: {method}")

def preprocess_for_model(
    image: Image.Image,
    model_type: str = "clip"
) -> torch.Tensor:
    """Preprocess image for various models"""
    if model_type == "clip":
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    elif model_type == "resnet":
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Default preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
    return preprocess(image)

def visualize_attention(
    image: Image.Image,
    attention_weights: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """Visualize attention weights on image"""
    # Ensure attention weights are 2D
    if len(attention_weights.shape) == 1:
        # Assume square attention map
        size = int(np.sqrt(attention_weights.shape[0]))
        attention_weights = attention_weights.reshape(size, size)
        
    # Resize attention to match image size
    attention_resized = cv2.resize(
        attention_weights,
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Normalize to 0-255
    attention_normalized = (attention_resized * 255).astype(np.uint8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on image
    image_array = np.array(image)
    overlay = cv2.addWeighted(image_array, 1-alpha, heatmap_rgb, alpha, 0)
    
    return Image.fromarray(overlay)
