import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from transformers import BertTokenizer
from model import MultiModalClassifier
from dataset import meme_Dataset  # optional if using shared transforms
import os
from transformers import ViTImageProcessor


NUM_CLASSES = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalClassifier(num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Mapping the labels
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive",
    3: "very_negative",
    4: "very_positive"
}

def predict_sentiment(image, text,extra_feats):
    image = image.convert("RGB")
    feature_extractor=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    # img_tensor = transform(image).unsqueeze(0).to(device)
    text_inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True)
    for key in text_inputs:
        text_inputs[key] = text_inputs[key].to(device)
    extra_feats=extra_feats.to(device)
    with torch.no_grad():
        outputs = model(img_tensor, text_inputs,extra_feats)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return label_map[predicted_class]

# Gradio UI
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Meme Text")
    ],
    outputs=gr.Label(label="Predicted Sentiment"),
    title="Meme Sentiment Classifier",
    description="Upload a meme image and enter its associated text to get sentiment prediction."
)

if __name__ == "__main__":
    interface.launch()
