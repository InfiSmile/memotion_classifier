import torch
import torch.nn as nn
from model import MultiModalClassifier
from dataset import meme_Dataset
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer,BertModel
import torch.optim as optim
import os
from sklearn.metrics import classification_report,f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# weights = torch.tensor(class_weights, dtype=torch.float).to(device)
SAVE_PATH = "checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_CLASSES = 5
NUM_EPOCHS = 10
LR = 5e-4

df= pd.read_csv(r'memotion_dataset_7k/labels.csv')
# sampled_df = df.groupby('overall_sentiment', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42)).reset_index(drop=True)
# max_count = df['overall_sentiment'].value_counts().min()

# sampled_df = df.groupby('overall_sentiment', group_keys=False).apply(
#     lambda x: x.sample(n=max_count, random_state=42)
# ).reset_index(drop=True)
sampled_df = df.sample(n=3000, random_state=42)
# n_per_class = 100  # Adjust based on smallest class
# sampled_df = df.groupby('overall_sentiment', group_keys=False).apply(
#     lambda x: x.sample(n=min(len(x), n_per_class), random_state=42)
# ).reset_index(drop=True)
# for col in ['humour', 'sarcasm', 'offensive', 'motivational']:
#     le = LabelEncoder()
#     sampled_df[col] = le.fit_transform(sampled_df[col])
#     print(dict(zip(le.classes_, le.transform(le.classes_))))

print(df['overall_sentiment'].value_counts())
label_encoder = LabelEncoder()
sampled_df['overall_sentiment'] = label_encoder.fit_transform(sampled_df['overall_sentiment'])
print(sampled_df['overall_sentiment'].unique())
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
train_df, test_df = train_test_split(sampled_df, test_size=0.2, random_state=42, stratify=sampled_df['overall_sentiment'])
# Compute class weights from training set
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['overall_sentiment']),
    y=train_df['overall_sentiment']
)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train=meme_Dataset(train_df,transform,tokenizer,r'memotion_dataset_7k\images')


train_loader=DataLoader(train, batch_size=8,shuffle=True)

test=meme_Dataset(test_df,transform,tokenizer,r'memotion_dataset_7k\images')
test_loader= DataLoader(test,batch_size=8,shuffle=False)

model = MultiModalClassifier(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_acc = 0.0
train_losses = []
test_losses = []
best_f1 = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss=0
    correct=0
    total=0

    pbar=tqdm(train_loader,desc=f'EPOCH {epoch+1}/{NUM_EPOCHS}')

    for img_tensor,text_input,label,extra_feats in pbar:
        extra_feats=extra_feats.to(device)
        img_tensor=img_tensor.to(device)
        label=label.to(device)

        for key in text_input:
            text_input[key] = text_input[key].to(device)

        outputs=model(img_tensor,text_input,extra_feats)

        loss=criterion(outputs,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
        preds=outputs.argmax(dim=1)
        correct+=(preds==label).sum().item()
        total+=label.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct/total)
        
    print(f"Epoch {epoch+1} Training avg Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")
    train_losses.append(epoch_loss/len(train_loader))
#------- Validation Loss-------------------------
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss=0
    test_epoch=0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar=tqdm(test_loader,desc=f'EPOCH {epoch+1}/{NUM_EPOCHS}')
        for img_tensor, text_inputs, labels,extra_feats in pbar:
            extra_feats=extra_feats.to(device)
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)

            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(device)

            outputs = model(img_tensor, text_inputs,extra_feats)
            loss=criterion(outputs,labels)
            test_loss+=loss.item()
            test_preds = outputs.argmax(dim=1)
            test_correct += (test_preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(test_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            
            pbar.set_postfix(loss=loss.item(), acc=test_correct/test_total)
    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch+1} complete — Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_correct/test_total:.4f}")
    # val_accuracy = test_correct / test_total
    
    test_losses.append(avg_test_loss)
    print("\nClassification Report:")
    print("F1 Score:", f1_score(all_labels, all_preds, average='weighted'))
    current_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    scheduler.step()
    if current_f1 > best_f1:
        best_f1=current_f1
        # best_acc = test_correct / test_total
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1
        }
        torch.save(checkpoint, os.path.join(SAVE_PATH, "best_model.pth"))
        print(f" Saved new best model with F1 score: {best_f1:.4f}")




plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")  
plt.show()