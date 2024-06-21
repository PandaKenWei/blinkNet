# 註釋待補
import torch
class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, num_epochs=10, save_path='model.pth'):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.save_path = save_path

    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, (video_frame, tags) in enumerate(self.dataloader):
                ### 確保數據在模型的相同裝置 ( CPU 或 GPU )
                video_frame = video_frame.to(device)
                tags = tags.to(device)

                # 轉換數據類行為 float
                video_frame = video_frame.float()

                self.optimizer.zero_grad()
                outputs = self.model(video_frame)

                """
                因為 torch.nn.CrossEntropyLoss
                輸入張量 ( 通常是模型的輸出 ) 應是一個二維張量 ( 2D tensor )，形狀為 [N, C]
                    - N: 一個 batch 的樣本數量
                    - C: 是輸出類別的種類數量
                目標張量 ( label ) 應是一個一維張量 ( 1D tensor )，形狀為 [N]
                """

                # 將 outputs 重塑為 [batch_size * time_steps, num_classes]
                batch_size, time_steps, num_classes = outputs.shape
                outputs = outputs.view(-1, num_classes)

                # 將 tags 重塑為 [batch_size * time_steps]
                tags = tags.view(-1)

                print(outputs.size())
                print(tags.size())

                loss = self.criterion(outputs, tags)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if batch_idx % 2 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()}')

            average_loss = total_loss / len(self.dataloader)
            print(f'Epoch {epoch+1}, Average Loss: {average_loss}')
    def save_model(self):
        """Saves the model to the specified path."""
        torch.save(self.model.state_dict(), self.save_path)
        print(f'Model saved to {self.save_path}')