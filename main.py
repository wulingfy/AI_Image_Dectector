import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
from huggingface_hub import hf_hub_download
from transformers import AutoModel


IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAPPING = {1: "Human", 0: "AI"}

MODEL_PATH = hf_hub_download(repo_id="Dafilab/ai-image-detector", filename="model_epoch_8_acc_0.9859.pth")


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 20),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = create_model('efficientnet_b4', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img)
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    return LABEL_MAPPING[predicted_class], confidence


image_path = 'IMG/PATH'
label, confidence = predict_image(image_path)

print(f'Author: {label}')
print(f'Percent: {confidence*100}%')
