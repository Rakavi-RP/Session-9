import torch
import gradio as gr
from model import ResNet50
from torchvision import transforms
from PIL import Image
import numpy as np
import json

with open('imagenet_classes.json', 'r') as f:
    class_labels = json.load(f)

device = torch.device('cpu')
model = ResNet50(num_classes=1000)
checkpoint = torch.load('best_model.pth', map_location=device)
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    return {class_labels[str(idx.item())]: prob.item() 
            for prob, idx in zip(top5_prob, top5_catid)}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="ImageNet Classifier",
    description="Upload an image to get top 5 predictions"
)

iface.launch()