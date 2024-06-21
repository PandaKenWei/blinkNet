from EyeBlinkDatasetAndPreprocessor import DataPreprocessor, EyeBlinkDataset
from model import BlinkNet, collate_fn_of_blink_net
from Trainer import Trainer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 數據加載與預處理物件
dataPreprocessor = DataPreprocessor()
data_paths = None
# 獲取訓練資料路徑
if not dataPreprocessor.get_status():
    print("錯誤")
else:
    data_paths = dataPreprocessor.get_video_and_annotation_paths()
# 使用字定義 Dataset 類並加載數據集
dataset = EyeBlinkDataset(data_paths)
# 定義資料集參數
batch_size = 2
shuffle = True
num_workers = 0 # 多少子進程來加載訓練資料
# 將 dataset 裝進 dataLoader 中
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_of_blink_net)
# 初始化模型
model = BlinkNet()
model.train() # 切換為訓練模式
# 優化器參數
learning_rate = 0.001
# 定義優化器
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# 定義損失函數
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.CrossEntropyLoss() # 多分類問題好像要用這個？

# 使用 Trainer 類進行訓練，需傳入模型、dataloader 與其他必要參數
trainer = Trainer(model, data_loader, criterion, optimizer, num_epochs=10)
trainer.train()
trainer.save_model()
