import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertModel
import torch
import matplotlib.pyplot as plt
from model import MultiModalClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import UnidentifiedImageError
from PIL import ImageFile
from transformers import ViTImageProcessor
#  ViTImageProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True

# print(labels.columns)

class meme_Dataset(Dataset):
    def __init__(self,dataframe,transform,tokenizer,img_dir):
        self.df=dataframe
        self.transform=transform
        self.tokenizer= tokenizer
        self.img_dir=img_dir
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.valid_indices = []
        # Initialize encoders for each extra feature
        self.humour_enc = LabelEncoder().fit(['funny', 'hilarious', 'not_funny', 'very_funny'])
        self.sarcasm_enc = LabelEncoder().fit(['general', 'not_sarcastic', 'twisted_meaning', 'very_twisted'])
        self.offensive_enc = LabelEncoder().fit(['hateful_offensive', 'not_offensive', 'slight', 'very_offensive'])
        self.motivation_enc = LabelEncoder().fit(['motivational', 'not_motivational'])

        for idx in range(len(self.df)):
            image_path = os.path.join(self.img_dir, self.df.iloc[idx]['image_name'])
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Check if it's a valid image
                self.valid_indices.append(idx)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Skipping corrupted image: {image_path} — {e}")

    def __getitem__(self,idx):
        idx = self.valid_indices[idx]
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row['image_name'])
        extra_feats = torch.tensor([
            self.humour_enc.transform([row['humour']])[0],
            self.sarcasm_enc.transform([row['sarcasm']])[0],
            self.offensive_enc.transform([row['offensive']])[0],
            self.motivation_enc.transform([row['motivational']])[0],
        ], dtype=torch.long)

        # row=self.df.iloc[idx]
        text=str(row['text_corrected'])
        # label=torch.tensor(row['overall_sentiment'])
        label = int(row['overall_sentiment'])         # must be int
        label = torch.tensor(label).long()    
        # image_name=row['image_name']
        # image_path= os.path.join(self.img_dir,image_name )
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        # img_tensor= self.transform(image)
        text_inputs=self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        return img_tensor,text_inputs,label,extra_feats
    
    def __len__(self):
        return len(self.valid_indices)


# bert = BertModel.from_pretrained("bert-base-uncased")






# for idx, (img_tensor, label, text_inputs) in enumerate(loader):
    
#     img = img_tensor[0].permute(1, 2, 0).numpy()  # Convert CHW -> HWC format for plt
#     plt.imshow(img)
#     plt.axis('off')  # Hide axes
#     plt.show()

  
#     print(f"Text: {text_inputs}")
    
   
#     print(f"Label: {label[0]}")

#     break














