import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import time
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 全局定义 collate_fn
def collate_fn(batch):
    images, input_ids, attention_masks, labels, ages, genders = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    ages = torch.stack(ages)
    genders = torch.stack(genders)
    labels = torch.stack(labels)
    return images, input_ids, attention_masks, labels, ages, genders

# 数据集类：归一化年龄 & 过采样少数类
class EyeDiseaseDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None, tokenizer=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        print(f"Loading Excel file from: {label_path}")
        self.df = pd.read_excel(label_path)
        print("Excel file loaded successfully.")
        
        self.image_files = set(os.listdir(image_dir))
        self.df = self.df[
            (self.df["Left-Fundus"].isin(self.image_files)) & 
            (self.df["Right-Fundus"].isin(self.image_files))
        ].reset_index(drop=True)
        
        self.class_labels = ["N", "D", "G", "C", "A", "H", "M", "O"]
        self.labels = self.df[self.class_labels].values
        self.left_fundus = self.df["Left-Fundus"].tolist()
        self.right_fundus = self.df["Right-Fundus"].tolist()

        # 归一化年龄到 [0, 1]
        age_min, age_max = self.df["Age"].min(), self.df["Age"].max()
        self.ages = (self.df["Age"].astype(float) - age_min) / (age_max - age_min)
        self.genders = self.df["Gender"].map({"Male": 0, "Female": 1}).astype(float).tolist()
        
        # 计算采样权重以处理类别不平衡
        class_counts = self.labels.sum(axis=0)
        self.sample_weights = 1.0 / (class_counts + 1e-6)  # 避免除以零
        self.sample_weights = self.sample_weights[self.labels.argmax(axis=1)]

        # 处理诊断关键词
        left_keyword_col = right_keyword_col = None
        for col in self.df.columns:
            if "left" in col.lower() and "keyword" in col.lower():
                left_keyword_col = col
            if "right" in col.lower() and "keyword" in col.lower():
                right_keyword_col = col
        self.left_keywords = self.df[left_keyword_col].fillna("No description").astype(str).tolist() if left_keyword_col else ["No description"] * len(self.df)
        self.right_keywords = self.df[right_keyword_col].fillna("No description").astype(str).tolist() if right_keyword_col else ["No description"] * len(self.df)

    def __getitem__(self, idx):
        left_img_path = os.path.join(self.image_dir, self.left_fundus[idx])
        right_img_path = os.path.join(self.image_dir, self.right_fundus[idx])
        left_img = Image.open(left_img_path).convert("RGB")
        right_img = Image.open(right_img_path).convert("RGB")
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        combined_img = torch.cat([left_img, right_img], dim=0)
        assert combined_img.shape == (6, 224, 224), f"图像形状错误: {combined_img.shape}"
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        combined_text = f"{self.left_keywords[idx]} {self.right_keywords[idx]}"
        tokens = self.tokenizer(combined_text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.float32)

        return combined_img, input_ids, attention_mask, label, age, gender

    def __len__(self):
        return len(self.df)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# U-Net 模型：添加辅助输出用于多尺度监督
class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=2):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = upconv_block(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = upconv_block(512, 256)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = upconv_block(256, 128)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = upconv_block(128, 64)
        self.dec1 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.aux_out = nn.Conv2d(128, out_channels, kernel_size=1)  # 辅助输出

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        aux_mask = self.aux_out(d2)  # 辅助掩码
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        return out, aux_mask

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Grad-CAM：支持 Grad-CAM++ 和左右眼分别计算
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx, device, is_gradcam_plus=False):
        self.model.eval()
        input_image = input_image.to(device)
        if not input_image.requires_grad:
            input_image = input_image.requires_grad_(True)

        batch_size = input_image.size(0)
        age = torch.zeros(batch_size, dtype=torch.float32).to(device)
        gender = torch.zeros(batch_size, dtype=torch.float32).to(device)

        outputs, _, _ = self.model(input_image, None, None, age, gender)
        self.model.zero_grad()
        score = outputs[0, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        if is_gradcam_plus:
            alpha = gradients.pow(2) / (2 * gradients.pow(2) + activations * gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
            weights = (alpha * torch.relu(gradients)).mean(dim=[2, 3], keepdim=True)
        else:
            weights = pooled_gradients
        
        cam = torch.zeros_like(activations, device=device)
        for i in range(activations.shape[1]):
            cam[:, i, :, :] = activations[:, i, :, :] * weights[:, i, :, :]
        cam = torch.mean(cam, dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam.unsqueeze(1)
        cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam

# 多模态模型：注意力融合 & 模态缺失处理
class MultimodalEyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(MultimodalEyeDiseaseModel, self).__init__()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        original_weights = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3, :, :] = original_weights
            self.resnet.conv1.weight[:, 3:, :, :] = original_weights
        self.resnet.fc = nn.Identity()
        
        self.unet = UNet(in_channels=6, out_channels=2)
        self.bert = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False, hidden_size=768)
        self.bert.eval()
        
        self.img_feature_dim = 2048
        self.text_feature_dim = self.bert.config.hidden_size  # 768，仅用于初始定义
        self.mask_feature_dim = 256
        
        self.modality_mask_img = nn.Parameter(torch.ones(self.img_feature_dim))
        self.modality_mask_text = nn.Parameter(torch.ones(self.text_feature_dim))
        self.modality_mask_mask = nn.Parameter(torch.ones(self.mask_feature_dim))
        
        # 添加文本特征投影层
        self.text_proj = nn.Linear(self.text_feature_dim, self.img_feature_dim)  # 768 -> 2048
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.img_feature_dim, num_heads=8)
        self.mask_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mask_fc = nn.Linear(2, self.mask_feature_dim)
        
        # 更新 combined_dim，使用投影后的维度
        combined_dim = self.img_feature_dim + self.img_feature_dim + self.mask_feature_dim + 2  # 2048 + 2048 + 256 + 2 = 4354
        self.attention = nn.MultiheadAttention(embed_dim=combined_dim, num_heads=2)
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids=None, attention_mask=None, age=None, gender=None):
        img_features = self.resnet(images)
        mask_pred, aux_mask = self.unet(images)
        mask_features = self.mask_pool(mask_pred).flatten(start_dim=1)  # [batch, 2]
        mask_features = self.mask_fc(mask_features)  # [batch, 256]
        
        img_features = img_features * self.modality_mask_img if images is not None else torch.zeros_like(img_features)
        if input_ids is not None and attention_mask is not None:
            with torch.no_grad():
                text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs[0][:, 0, :]  # [batch, 768]
            text_features = text_features * self.modality_mask_text
        else:
            text_features = torch.zeros(images.size(0), self.text_feature_dim, device=images.device)
        
        mask_features = mask_features * self.modality_mask_mask
        
        # 投影文本特征到图像特征维度
        text_features = self.text_proj(text_features)  # [batch, 768] -> [batch, 2048]
        img_features, _ = self.cross_attention(img_features.unsqueeze(0), text_features.unsqueeze(0), text_features.unsqueeze(0))
        img_features = img_features.squeeze(0)
        
        age = age.unsqueeze(1) if age is not None else torch.zeros(images.size(0), 1, device=images.device)
        gender = gender.unsqueeze(1) if gender is not None else torch.zeros(images.size(0), 1, device=images.device)
        
        combined = torch.cat((img_features, text_features, mask_features, age, gender), dim=1)  # [batch, 4354]
        combined, _ = self.attention(combined.unsqueeze(0), combined.unsqueeze(0), combined.unsqueeze(0))
        combined = combined.squeeze(0)
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, img_features, (mask_pred, aux_mask)

# 平滑性损失
def smoothness_loss(mask):
    diff_x = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    diff_y = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
    return diff_x.mean() + diff_y.mean()

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_root = "D:\\Project\\ZiJin\\a07\\a07-code\\images"
    label_path = "D:\\Project\\ZiJin\\a07\\a07-code\\labels.xlsx"
    
    train_dataset = EyeDiseaseDataset(os.path.join(dataset_root, "train"), label_path, transform, tokenizer)
    val_dataset = EyeDiseaseDataset(os.path.join(dataset_root, "validation"), label_path, transform, tokenizer)
    test_dataset = EyeDiseaseDataset(os.path.join(dataset_root, "test"), label_path, transform, tokenizer)
    
    sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=True)
    
    model = MultimodalEyeDiseaseModel(num_classes=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    criterion_seg = DiceLoss()
    optimizer = torch.optim.Adam([
        {"params": model.resnet.parameters(), "lr": 0.0001},
        {"params": model.unet.parameters(), "lr": 0.0001},
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": model.fc1.parameters(), "lr": 0.0001},
        {"params": model.fc2.parameters(), "lr": 0.0001},
        {"params": model.mask_fc.parameters(), "lr": 0.0001},
        {"params": model.text_proj.parameters(), "lr": 0.0001}
    ])
    
    writer = SummaryWriter('runs/multimodal_eye_disease_' + time.strftime("%Y%m%d-%H%M%S"))
    grad_cam = GradCAM(model, model.resnet.layer4[-1])
    
    def train_model(model, criterion_cls, criterion_seg, optimizer, train_loader, val_loader, num_epochs, patience=15, warmup_epochs=5):
        model.train()
        scaler = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        best_val_loss = float('inf')
        best_model_path = "best_multimodal_eye_disease_model.pth"
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            running_loss = running_seg_loss = 0.0
            for i, (images, input_ids, attention_masks, labels, ages, genders) in enumerate(train_loader):
                images, input_ids, attention_masks, ages, genders, labels = [x.to(device) for x in [images, input_ids, attention_masks, ages, genders, labels]]
                
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs, _, (mask_pred, aux_mask) = model(images, input_ids, attention_masks, ages, genders)
                    loss_cls = criterion_cls(outputs, labels)
                    
                    pseudo_masks = torch.zeros_like(mask_pred)
                    if epoch >= warmup_epochs:
                        model.eval()
                        with torch.no_grad():
                            for b in range(images.size(0)):
                                class_idx = torch.argmax(labels[b]).item()
                                left_eye = images[b:b+1, :3, :, :]
                                right_eye = images[b:b+1, 3:, :, :]
                                left_input = torch.cat([left_eye, torch.zeros_like(left_eye)], dim=1)
                                right_input = torch.cat([torch.zeros_like(right_eye), right_eye], dim=1)
                                with torch.enable_grad():
                                    cam_left = grad_cam.generate(left_input, class_idx, device, is_gradcam_plus=True)
                                    cam_right = grad_cam.generate(right_input, class_idx, device, is_gradcam_plus=True)
                                pseudo_masks[b, 0] = cam_left[0]
                                pseudo_masks[b, 1] = cam_right[0]
                        model.train()
                        loss_seg = criterion_seg(mask_pred, pseudo_masks)
                        aux_mask_resized = nn.functional.interpolate(aux_mask, size=(224, 224), mode='bilinear', align_corners=False)
                        loss_seg_aux = criterion_seg(aux_mask_resized, pseudo_masks)
                        loss_smooth = smoothness_loss(mask_pred)
                        loss_seg_total = loss_seg + 0.5 * loss_seg_aux + 0.1 * loss_smooth
                    else:
                        loss_seg_total = torch.tensor(0.0).to(device)
                    
                    loss = loss_cls + (0.3 * loss_seg_total if epoch >= warmup_epochs else 0.0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss_cls.item()
                running_seg_loss += loss_seg_total.item() if epoch >= warmup_epochs else 0.0
                
                writer.add_scalar('训练损失/分类/批次', loss_cls.item(), epoch * len(train_loader) + i)
                if epoch >= warmup_epochs:
                    writer.add_scalar('训练损失/分割/批次', loss_seg_total.item(), epoch * len(train_loader) + i)
            
            epoch_loss = running_loss / len(train_loader)
            epoch_seg_loss = running_seg_loss / len(train_loader) if epoch >= warmup_epochs else 0.0
            print(f"周期 {epoch+1}/{num_epochs}, 分类损失: {epoch_loss:.4f}, 分割损失: {epoch_seg_loss:.4f}")
            
            model.eval()
            val_loss = val_seg_loss = 0.0
            with torch.no_grad():
                for images, input_ids, attention_masks, labels, ages, genders in val_loader:
                    images, input_ids, attention_masks, ages, genders, labels = [x.to(device) for x in [images, input_ids, attention_masks, ages, genders, labels]]
                    outputs, _, (mask_pred, aux_mask) = model(images, input_ids, attention_masks, ages, genders)
                    loss_cls = criterion_cls(outputs, labels)
                    
                    if epoch >= warmup_epochs:
                        pseudo_masks = torch.zeros_like(mask_pred)
                        for b in range(images.size(0)):
                            class_idx = torch.argmax(labels[b]).item()
                            left_eye = images[b:b+1, :3, :, :]
                            right_eye = images[b:b+1, 3:, :, :]
                            left_input = torch.cat([left_eye, torch.zeros_like(left_eye)], dim=1)
                            right_input = torch.cat([torch.zeros_like(right_eye), right_eye], dim=1)
                            with torch.enable_grad():
                                cam_left = grad_cam.generate(left_input, class_idx, device, is_gradcam_plus=True)
                                cam_right = grad_cam.generate(right_input, class_idx, device, is_gradcam_plus=True)
                            pseudo_masks[b, 0] = cam_left[0]
                            pseudo_masks[b, 1] = cam_right[0]
                        loss_seg = criterion_seg(mask_pred, pseudo_masks)
                    else:
                        loss_seg = torch.tensor(0.0).to(device)
                    
                    val_loss += loss_cls.item()
                    val_seg_loss += loss_seg.item() if epoch >= warmup_epochs else 0.0
            
            val_loss = val_loss / len(val_loader)
            val_seg_loss = val_seg_loss / len(val_loader) if epoch >= warmup_epochs else 0.0
            print(f"周期 {epoch+1}, 验证分类损失: {val_loss:.4f}, 验证分割损失: {val_seg_loss:.4f}")
            
            total_val_loss = val_loss + (0.3 * val_seg_loss if epoch >= warmup_epochs else 0.0)
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"保存最优模型: {best_model_path}, 总验证损失: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"验证损失未改善，当前计数: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print("触发早停，停止训练！")
                    break
            
            scheduler.step()
            model.train()
        
        return best_model_path
    
    best_model_path = train_model(model, criterion_cls, criterion_seg, optimizer, train_loader, val_loader, num_epochs=10, patience=5, warmup_epochs=5)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    def evaluate(model, test_loader):
        model.eval()
        predictions, actuals, all_images, all_features, all_mask_preds = [], [], [], [], []
        with torch.no_grad():
            for images, input_ids, attention_masks, labels, ages, genders in test_loader:
                images, input_ids, attention_masks, ages, genders, labels = [x.to(device) for x in [images, input_ids, attention_masks, ages, genders, labels]]
                outputs, features, (mask_pred, _) = model(images, input_ids, attention_masks, ages, genders)
                outputs = torch.sigmoid(outputs)
                mask_pred = torch.sigmoid(mask_pred)
                predictions.append(outputs.cpu())
                actuals.append(labels.cpu())
                all_images.append(images.cpu())
                all_features.append(features.cpu())
                all_mask_preds.append(mask_pred.cpu())
                
                # Grad-CAM 可视化（只对第一个样本）
                with torch.enable_grad():
                    input_image = images[:1].clone().detach().requires_grad_(True)  # 确保梯度追踪
                    cam = grad_cam.generate(input_image, torch.argmax(labels[0]).item(), device, is_gradcam_plus=True)
                    cam = cam[0, 0].detach.cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 规范化
                    heatmap = plt.cm.jet(cam)[:, :, :3]
        
        predictions = torch.cat(predictions)
        actuals = torch.cat(actuals)
        all_images = torch.cat(all_images)
        all_features = torch.cat(all_features)
        all_mask_preds = torch.cat(all_mask_preds)
        
        pred_binary = (predictions > 0.5).float()
        accuracy = (pred_binary == actuals).float().mean().item()
        
        true_positives = (pred_binary * actuals).sum(dim=0)
        predicted_positives = pred_binary.sum(dim=0)
        actual_positives = actuals.sum(dim=0)
        precision_per_class = torch.where(predicted_positives > 0, true_positives / predicted_positives, torch.zeros_like(true_positives))
        avg_precision = precision_per_class.sum() / max((predicted_positives > 0).sum(), 1)
        recall_per_class = torch.where(actual_positives > 0, true_positives / actual_positives, torch.zeros_like(true_positives))
        avg_recall = recall_per_class.sum() / max((actual_positives > 0).sum(), 1)
        f1_per_class = torch.where((precision_per_class + recall_per_class) > 0, 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class), torch.zeros_like(precision_per_class))
        avg_f1 = f1_per_class.sum() / max((actual_positives > 0).sum(), 1)
        
        # 可视化
        left_image = all_images[0][:3].permute(1, 2, 0).numpy()
        left_image = (left_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        left_mask_pred = (all_mask_preds[0][0] > 0.5).float().numpy()
        left_image_with_mask = left_image.copy()
        left_image_with_mask[left_mask_pred == 1] = [1, 0, 0]
        left_image_with_cam = left_image * 0.5 + heatmap * 0.5
        
        writer.add_image('测试/输入图像/左眼', left_image, global_step=0, dataformats='HWC')
        writer.add_image('测试/叠加掩码/左眼', left_image_with_mask, global_step=0, dataformats='HWC')
        writer.add_image('测试/Grad-CAM/左眼', left_image_with_cam, global_step=0, dataformats='HWC')
        
        # 指标记录
        class_labels = ["N", "D", "G", "C", "A", "H", "M", "O"]
        print(f"\n测试集准确率: {accuracy * 100:.2f}%")
        print("\n每类精确率:")
        for label, precision in zip(class_labels, precision_per_class):
            print(f"{label}: {precision.item()*100:.2f}%")
        print(f"平均精确率: {avg_precision.item()*100:.2f}%")
        print("\n每类召回率:")
        for label, recall in zip(class_labels, recall_per_class):
            print(f"{label}: {recall.item()*100:.2f}%")
        print(f"平均召回率: {avg_recall.item()*100:.2f}%")
        print("\n每类F1分数:")
        for label, f1 in zip(class_labels, f1_per_class):
            print(f"{label}: {f1.item()*100:.2f}%")
        print(f"平均F1分数: {avg_f1.item()*100:.2f}%")
        
        writer.add_scalar('测试/准确率', accuracy, 0)
        writer.add_scalar('测试/平均精确率', avg_precision.item(), 0)
        writer.add_scalar('测试/平均召回率', avg_recall.item(), 0)
        writer.add_scalar('测试/平均F1分数', avg_f1.item(), 0)
        
        aps = []
        for i, label in enumerate(class_labels):
            precision, recall, _ = precision_recall_curve(actuals[:, i].numpy(), predictions[:, i].numpy())
            ap = np.trapz(precision, recall)
            aps.append(ap)
            writer.add_pr_curve(f'PR曲线/{label}', actuals[:, i], predictions[:, i], global_step=0)
        mAP = np.mean(aps)
        writer.add_scalar('测试/mAP', mAP, 0)
        print(f"mAP: {mAP*100:.2f}%")
        
        cm = confusion_matrix(actuals.flatten(), pred_binary.flatten())
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        writer.add_figure('混淆矩阵', fig, global_step=0)
        
        writer.close()
        
        return (predictions, actuals, precision_per_class, avg_precision, recall_per_class, avg_recall, 
                f1_per_class, avg_f1, all_images, all_features, all_mask_preds)
    
    evaluate(model, test_loader)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.multiprocessing.freeze_support()
    main()