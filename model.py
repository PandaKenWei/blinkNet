import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

# CNN 特徵提取器
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # 第一層卷積層，輸入通道數為 3（RGB圖像），輸出通道數為 16，卷積核大小為 3x3，邊界填充為1
        # N*3*480*640 -> N*16*480*640
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 第一層批次標準化
        # N*16*480*640
        self.bn1 = nn.BatchNorm2d(16)
        # 第二層卷積層，輸入通道數為16，輸出通道數為32，卷積核大小為3x3，邊界填充為1
        # N*16*480*640 -> N*32*480*640
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 第二層批次標準化
        # N*32*480*640
        self.bn2 = nn.BatchNorm2d(32)
        # 第三層卷積層，輸入通道數為32，輸出通道數為64，卷積核大小為3x3，邊界填充為1
        # N*32*480*640 -> N*64*480*640
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 第三層批次標準化
        # N*64*480*640
        self.bn3 = nn.BatchNorm2d(64)
        # 最大池化層，卷積核大小為2x2，步長為2
        # N*64*480*640 -> N*64*240*320
        self.pool = nn.MaxPool2d(2, 2)
        # 全局平均池化層，每個單獨的「通道」都會被壓縮成指定的大小
        # N*64*240*320 -> N*64*1*1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通過第一卷積層與標準化，然後應用 Leaky ReLU 激活函數，接著進行池化
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        # 通過第二卷積層...
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        # 通過第三卷積層...
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        # 通過全局平均池化層
        x = self.global_avg_pool(x)
        # 展平特徵(保留通道數)，為後續的連接層準備
        # N*64*1*1 -> N*64
        x = x.view(x.size(0), -1)
        return x
    
# TCN 模型
class Chomp1d(nn.Module):
    """
    Chomp1d 這是一個輔助 class
    其主要功能是移除因為膨脹卷積(dilated convolution)而引入的多餘的填充，保證輸出尺寸與輸入尺寸相同
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 取得 padding 的數量
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        使用陣列切片，然後將右測捕的 padding 所產生的「未來值」給去除，以達到
        1. 輸入與輸出相等
        2. 不參考未來數據 ( 雖然是 0 )
        然後使用 .coutuguous() 來保證數據的記憶體是連續空間
        """
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    """
    TemporalBlock 這個類代表了 TCN 中的一個基本塊
    通常包含兩個卷積層，每個卷積層後面都跟有非線性激活、正規化和 dropout
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 定義第一層捲積層 ( Conv1d ) 並使用 weight_norm 正則化使得收斂更快
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 定義第一層「去除層」，用來去除膨脹後的 output 維度與 input 不一致
        self.chomp1 = Chomp1d(padding)
        # 激勵函數與 dropout
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 定義第二層捲積層、去除層、激勵函數與 dropout
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # 封裝模型過程，供 forward 函式調用
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        """
        定義殘差連接的邏輯
        如果 input 與 output 的通道數不相同，就使用 kernal size 為 1*1 的捲積核進行通道數改變
        反之則該層定義為 None
        """
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.relu = nn.ReLU()
        # 初始化權重的函數
        self.init_weights()

    def init_weights(self):
        # 將第一捲積層的初始化均值設為 0 並初始化標準差設為 0.01 的正態分布
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        """
        定義殘差連接層
        如果相 input 與 output 相等，則恆等映射
        反之如果不相等 ( downsample 層不為 None ) 則定義一個下採樣轉換層
        """
        res = x if self.downsample is None else self.downsample(x)
        # 殘差連接相加
        return self.relu(out + res)
    
class TemporalConvNet(nn.Module):
    """
    TemporalConvNet 這是 TCN 模型的主要部分
    封裝了多個 TemporalBlock 層來形成整個網絡
    模型可以擴展到多個層次，每個層次可以有不同的設定 ( 如通道數、膨脹因子等 )

    Args:
        num_inputs (int): 輸入的通道數
        num_channels (lsit[int]): 每個隱藏層的通道數量，例如 [25, 25, 25, 25] 表示有 4 個隱藏層，每個隱藏藏都有 25 個通道
        kernal_size (int): 捲積核大小，默認大小為 2
        dropout (float): 隨機丟棄比例，介於 0 - 1 之間，默認為 0.2

    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        # 初始化「層」陣列，用來儲存多個 TemporalBlock
        layers = []
        num_levels = len(num_channels)
        """
        使用迴圈來建立每一個 TemporalBlock
        
        Args:
            dilation_size (int): 膨脹因子，隨著層的變深，程指數增加。4 層就是 1 -> 2 -> 4 -> 8
            in_channels (int): 每層的輸入通道數，第一層為 num_inputs，後續每一層都是指定的通道數
            out_channels (int): 輸出通道數，為指定的通道數。結合 in_channels 來看，就是除了第一層之外，後續每一層的輸入通道數都是前一層的輸出通道數。
        """
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            """
            使用每層定義的不同參數，將多層 TemporalBlck 增加到 layers 陣列中
            其中 padding 的數量，因為「膨脹」的關係，要定義為 (k-1)*d 才能符合需求
            """
            layers.append( TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size-1) * dilation_size, dropout=dropout))
        # 將每層 TemporalBlck 封裝程模型過程，供 forward 函數使用
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: size of (Batch, input_channel, window size)
        """
        return self.network(x)

# 整合 CNN 和 TCN 的模型
class BlinkNet(nn.Module):
    def __init__(self):
        super(BlinkNet, self).__init__()
        # CNN 提取器，input 為 N*3*480*640，output 為 N*64
        self.cnn = CNNFeatureExtractor()
        # TCN 模型，input 需要銜接 CNN 提取器的一為特徵數據 64，output 為 128
        self.tcn = TemporalConvNet(num_inputs=64, num_channels=[64, 64, 128, 128])
        # 最終輸出 3 個類別，分別對應每個 window 的三種狀態:
        #   - 0: 非 1 及 2 的狀態
        #   - 1: 開始 / 結束 眨眼的那一幀
        #   - 2: 正在眨眼的過程
        self.fc = nn.Linear(128, 3) 

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, T, H, W, C)
                - B: Batch size
                - T: Timesteps, 一個 window 內包含的幀 ( frames ) 數量
                - H: 圖片高
                - W: 圖片寬
                - C: 通道數
        """

        batch_size, timesteps, H, W, C = x.size()
        # 重塑 x 以匹配 CNN 的輸入需求 (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size * timesteps, C, H, W)
        # CNN 特徵提取，Shape: (B*T, 64)
        cnn_out = self.cnn(x)
        # 重新分割特徵以符合時間序列，其中 -1 表示自動計算維度，Shape: (B, T, 64)
        cnn_out = cnn_out.view(batch_size, timesteps, -1)
        # 將特徵維度移到第二維，以符合 TCN 的輸入要求，形狀為 (B, C, T)
        # # TCN 處理，Shape: (B, C, T)
        tcn_out = self.tcn(cnn_out.transpose(1, 2))

        # 取最後一個時間點的輸出來進行預測，形狀為 (B, C)
        # out = self.fc(t_out[:, :, -1])
        # FC 層處理 window 中所有 frame 的輸出，Shape: (B, T, 3)
        out = self.fc(tcn_out.transpose(1, 2))

        return out
    
def collate_fn_of_blink_net(batch):
    """
    自定義一個方法來傳入 DataLoader 類，用來處理非常規的數據來組合成 batch
    
    Args:
        Dataset 的 __getitem__ 返回的是: (frames, labels)
            - frames: 是一個 window 的數個 frame 組成的 list
            - labels: 是一個 window 的數個 label 組成的 list
    Returns:
        torch 張量的 tuple: (batch_frames, batch_labels)
            - batch_frames (Tensor): (B, per window num frames, C, H, W)
            - batch_labels (Tensor): (B, per window num labels)
    """
    
    # 初始化兩個列表來收集所有樣本的框架和標籤
    batch_frames = []
    batch_labels = []

    # 遍歷從 DataLoader 來的每個樣本
    for frames, labels in batch:
        # 將每個樣本的 frame 從 NumPy list 轉換為 PyTorch 的張量 ( tensor ) -> (per window num frames, C, H, W)
        frames_tensor = torch.stack([torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame for frame in frames])
        # 將每個樣本的 label 轉換張量，並指定數據類型為 long
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # 將一個 batch 中的個別「數據」組合成一個 list
        batch_frames.append(frames_tensor) # list[(per window num frames, C, H, W)]
        batch_labels.append(labels_tensor) # list[(per window num labels)]

    # 使用 torch.stack 將一個 batch 也壓進張量中
    batch_frames = torch.stack(batch_frames)  # 最終為五維張量 -> (B, per window num frames, C, H, W)
    batch_labels = torch.stack(batch_labels)  # 最終為二維張量 -> (B, per window num labels)

    return batch_frames, batch_labels