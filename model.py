import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50
from transformers import ViTModel, ViTFeatureExtractor
class MultiModalClassifier(nn.Module):
    def __init__(self,num_classes ):
        super().__init__()

        # base_resnet=resnet50(weights='ResNet50_Weights.DEFAULT')
        self.encoder_image = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # self.encoder_image = nn.Sequential(*list(base_resnet.children())[:-1])
        self.img_fc = nn.Linear(768, 512) #2048 channels on the last conv layer of resnet

        self.encoder_text=BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, 512)
        self.humour_embed = nn.Embedding(num_embeddings=4, embedding_dim=8)      # funny, hilarious, etc.
        self.sarcasm_embed = nn.Embedding(num_embeddings=4, embedding_dim=8)     # general, twisted, etc.
        self.offensive_embed = nn.Embedding(num_embeddings=4, embedding_dim=8)   # hateful, slight, etc.
        self.motiv_embed = nn.Embedding(num_embeddings=2, embedding_dim=8)       # motivational, not
        

        self.classifier = nn.Sequential(
            nn.LayerNorm(1152),
            nn.Dropout(0.3),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self.fc_extra = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU()
        )


    def forward(self,image,text,extra_feats):
        encoded_img = self.encoder_image(pixel_values=image).last_hidden_state
        encoded_img = encoded_img[:, 0]
        # encoded_img=self.encoder_image(image).squeeze(-1).squeeze(-1)
        encoded_img=self.img_fc(encoded_img)
        
        encoded_txt= self.encoder_text(**text)
        encoded_txt=self.text_fc(encoded_txt.last_hidden_state[:, 0])


        # Embed each feature
        humour_emb = self.humour_embed(extra_feats[:, 0])
        sarcasm_emb = self.sarcasm_embed(extra_feats[:, 1])
        offensive_emb = self.offensive_embed(extra_feats[:, 2])
        motiv_emb = self.motiv_embed(extra_feats[:, 3])

        # Concatenate and transform
        extra_emb = torch.cat([humour_emb, sarcasm_emb, offensive_emb, motiv_emb], dim=1)
        extra_feat_vector = self.fc_extra(extra_emb)

        combined=torch.cat([encoded_img,encoded_txt,extra_feat_vector],dim=1)

        return self.classifier(combined)
    










        
        
        
        

