import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel

def load_image(image_path, max_num=12):
    """
    Load and preprocess an image for the model.
    
    Args:
        image_path (str): Path to the image file
        max_num (int): Maximum number of patches to return
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image)
    
    # Add batch dimension and repeat if needed
    image_tensor = image_tensor.unsqueeze(0)
    if max_num > 1:
        image_tensor = image_tensor.repeat(max_num, 1, 1, 1)
    
    return image_tensor

def split_model(model_path):
    """
    Determine the device mapping for model splitting based on available GPUs.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        dict: Device mapping configuration
    """
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        return "cpu"
    elif num_gpus == 1:
        return "cuda:0"
    else:
        # For multiple GPUs, create a balanced device map
        device_map = {}
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        num_layers = len(model.encoder.layer) if hasattr(model, 'encoder') else len(model.layers)
        
        layers_per_gpu = num_layers // num_gpus
        for i in range(num_gpus):
            start_idx = i * layers_per_gpu
            end_idx = start_idx + layers_per_gpu if i < num_gpus - 1 else num_layers
            device_map[f"encoder.layer.{start_idx}"] = f"cuda:{i}"
            device_map[f"encoder.layer.{end_idx-1}"] = f"cuda:{i}"
        
        return device_map 