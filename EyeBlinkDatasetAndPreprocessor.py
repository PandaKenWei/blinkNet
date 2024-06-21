import os
from tkinter import Tk, messagebox
from tkinter.filedialog import askdirectory
import torch
from torch.utils.data import Dataset
import cv2
from typing import Tuple
# 進度條函式庫
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self):
        """
        初始化 DataPreprocessor 類，並提示用戶選擇一個資料夾。
        """
        self._data_folder_path = self._select_folder()

    def _select_folder(self) -> str:
        """
        使用 tkinter 創建一個文件選擇對話框，讓用戶選擇一個資料夾，並驗證資料夾結構是否符合要求。
        
        Returns:
        str: 用戶選擇的資料夾的完整路徑，如果用戶選擇取消則返回 None。
        """
        root = Tk()
        root.withdraw()  # 隱藏 Tkinter 的主窗口
        
        # 初始化驗證變數和重新選擇變數
        folder_selected = False
        retry_select = True
        
        while folder_selected ^ retry_select:
            folder_path = askdirectory(parent=root, title="選擇一個資料夾")
            # 如果選擇的目標資料夾通過規則檢驗，修改驗證變數為 True
            if self._is_folder_structure_valid(folder_path):
                folder_selected = True
                continue
            
            # 反之，則提示使用者錯誤信息並詢問是否重新選擇資料夾
            retry_select = self._show_warning_and_is_re_select("選擇的資料夾結構不符合要求，請重新選擇或取消。")

        root.destroy()
        return folder_path if folder_path != '' else None
    
    def _is_folder_structure_valid(self, folder_path:str) -> bool:
        """
        檢查資料夾是否包含所需的結構。

        Args:
        folder_path (str): 需要檢查的資料夾路徑。

        Returns:
        bool: 如果資料夾包含所需結構，返回 True; 否則返回 False。
        """
        
        # 檢查路徑是否為 None
        if folder_path is None:
            return False

        # 檢查目標資料夾路徑是否合法
        if not os.path.exists(folder_path):
            return False
        
        """# 檢查目標資料夾結構是否合法
        目標資料夾需要包含:
            1. n 個子資料夾，但不能有含其他檔案
            2. 個子資料夾需要有一個 .avi 和一個 .txt，且不能有其他檔案或資料夾
            3. 檢查資料夾內僅包含子資料夾並無其他檔案
        """
        # 遍歷目標資料夾
        for item in os.listdir(folder_path):
            # 取每個子資料夾的路徑
            item_path = os.path.join(folder_path, item)
            # 如果有非資料夾的檔案，返回 False
            if not os.path.isdir(item_path):
                return False

            files = os.listdir(item_path);
            # 如果子資料夾內容不等於 2 則不符合條件
            if len(files) != 2:
                return False
            
            # 讓名稱進行排序，這樣 .avi 會在 .txt 前面
            files_sorted = sorted(files)
            # 如果第一個檔案的副檔名不是 .avi 則不符合條件
            if not files_sorted[0].endswith('.avi'):
                return False
            # 如果第二個檔案的副檔名不是 .txt 則不符合條件
            if not files_sorted[1].endswith('.txt'):
                return False

        # 所有檢查都通過
        return True
    
    def _show_warning_and_is_re_select(self, message:str, title:str = None) -> bool:
        """
        顯示一個警告對話框。

        Args:
        message (str): 顯示的消息內容。
        title (str): 對話框的標題。

        Returns:
        bool: 如果用戶選擇重試則返回 True，如果選擇取消則返回 False。
        """
        root = Tk()
        root.withdraw()

        if title is None:
            title = "選擇的目標資料夾有問題"

        return messagebox.askretrycancel(title, message)

    def get_video_and_annotation_paths(self) -> list:
        """
        遍歷目標資料夾，尋找所有的 .avi 視頻文件及其對應的 .txt 標記文件。
        
        Returns:
        list[tuple]: 一個元組字典， key 為流水號(從 0 開始)， value 為相應的 .avi 檔案的完整路徑和 .txt 標記文件的完整路徑的 tuple。
        """

        # 初始化返回的結構變數
        video_annotation_pairs = []
        # 初始化流水號(從0開始)
        index = 0
        # 遍歷整個目標資料夾
        for folder_name in os.listdir(self._data_folder_path):
            # 子資料夾的完整路徑
            subfolder_path = os.path.join(self._data_folder_path, folder_name)
            # 取得子資料夾的所有檔案並依照英文排序
            files = sorted(os.listdir(subfolder_path))
            # 取得 .avi 檔案和 .txt 檔案的完整路徑
            avi_path = os.path.join(subfolder_path, files[0])
            txt_path = os.path.join(subfolder_path, files[1])
            # 將路徑包成元組存進字典中
            video_annotation_pairs.append((avi_path, txt_path))

            index += 1
        return video_annotation_pairs
    
    def get_status(self) -> bool:
        """
        查看資料前處理是否正常

        Return:
        bool: 正常回傳 True 反之不正常就回傳 False
        """
        
        status = True
        if self._data_folder_path is None:
            status = False

        return status

class EyeBlinkDataset(Dataset):
    def __init__(self, data_paths: list, window_size: int = 70, stride: int = 20) -> None:
        """
        初始化數據集
        Args:
            data_paths(list of tuple): 包含每組影片檔案與標籤的檔案路徑
            window_size(int): 每個 window 的幀數
            stride(int): 移動窗口間的 frame 數
        """
        self.video_paths = [path[0] for path in data_paths]
        self.frame_labels_path = [path[1] for path in data_paths]
        self.window_size = window_size
        self.stride = stride
        # 準備每支影片的 cv 處理器
        self.video_caps = [cv2.VideoCapture(p) for p in self.video_paths]
        # 準備每支影片的標籤 -> list[list[tuple(frame_index, blink_status)]]
        self.frame_labels = [self._prepare_label(path) for path in self.frame_labels_path]
        # 儲存每支影片的可執行 window 數量 -> list[list[tuple]]
        self.window_indices = self._calculate_windows()
    
    def __len__(self) -> int:
        """
        返回所有影片的可執行 window 數量: 
        1. 為避免跨影片的 window，每個影片的可執行 window 要獨立計算
        2. 計算公式: 單影片總 frame 除 步長，最後補上不滿足一個布長的次數 1
        """
        total_windows = 0
        for cap in self.video_caps:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - self.window_size + self.stride # 計算 windiow 數量需要扣掉一個 window 大小並加上一個 stride 的 frame 量
            if frame_count >= self.window_size:
                total_windows += frame_count // self.stride + 1
        return total_windows - 1000 if total_windows > 1000 else total_windows
    
    def __getitem__(self, index) -> tuple:
        """
        根據給定的索引返回一「window」的幀及其標記
        Args:
            param index: 執行的 window 的索引
        """

        # 找到對應的影片尼其內部的起始幀索引
        video_idx = 0
        ### 在每個影片中計算可能的 window 的起始點
        window_account = 0 # 累積 window 計數，確定 idx 屬於哪個影片
        # 初始化 start_frame_idx 和 end_frame_idx
        start_frame_idx = None
        end_frame_idx = None

        ### 確定每次呼叫的 index 屬於哪個影片，並獲取其內部的起始索引
        for video_idx, windows_per_video in  enumerate(self.window_indices):
            if index < window_account + len(windows_per_video):
                window_idx = index - window_account
                start_frame_idx, end_frame_idx = windows_per_video[window_idx]
                break
            window_account += len(windows_per_video)

        # 防呆
        if start_frame_idx is None or end_frame_idx is None:
            raise IndexError("取得 Window 的起始 frame idx 發生錯誤")
        
        ### window 中的所有幀與 label 的 list 變數
        frames = []
        labels = []        

        # 根據影片索引讀取特定的影片
        cap = self.video_caps[video_idx]
        ### 並取得特定「Window」的所有幀與標籤
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx) # 將影片位置預設為起始幀
        for i in range(end_frame_idx - start_frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                 print(f"第 {start_frame_idx + i} 幀讀取失敗")
                 continue
            # 添加 frame
            frames.append(frame)
            # 要抓取 label，因此要計算單個 window 走到哪一個 frame 了
            frame_idx = start_frame_idx + len(frames) - 1
            # 添加對應的標籤
            labels.append(self.frame_labels[video_idx][frame_idx][1]) #  if frame_idx < len(self.frame_labels[video_idx]) else 0

        # 返回「window」中幀及其對應的標記
        return (frames, labels)

    def __del__(self) -> None:
        """
        確保釋放所有影片檔案
        """
        for cap in self.video_caps:
            cap.release()
    
    def _prepare_label(self, label_path) -> list:
        """
        解析標籤文件，返回每幀的標籤列表
        
        return:
            list[tuple(frame_index, blink_status)]
        """
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                frame_index = int(parts[0])
                # 檢查是否有眨眼狀態標記
                blink_status = int(parts[2]) if len(parts) > 2 else 0
                labels.append((frame_index, blink_status))
        return labels

    def _calculate_windows(self) -> list:
        """
        計算每個影片可以產生的可執行 window 數量，並返回相當於影片數量的 list -> 
        其中 list 儲存了每支影片的每個 window 的開始與結束 idx -> [(start_idx, end_idx)]

        return:
            list[list[tuple]]
        """

        windows_list_per_video = []
        for cap in self.video_caps:
            # 取得單支影片的總 frame 數
            total_frame_per_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 單支影片的每一可執行 window 的起始與結束 frame idx 的 tuple list
            windows_per_video = []
            for i in range(0, total_frame_per_video - 1 - self.window_size + self.stride, self.stride):
                start_frame_idx = i
                # 要確保最後一個 window 的結束幀不會超過該影片的最後一幀
                end_frame_idx = i + self.window_size - 1 if i + self.window_size - 1 < total_frame_per_video else total_frame_per_video - 1
                windows_per_video.append((start_frame_idx, end_frame_idx))
            windows_list_per_video.append(windows_per_video)

        return windows_list_per_video

