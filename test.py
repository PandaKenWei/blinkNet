from EyeBlinkDatasetAndPreprocessor import DataPreprocessor

dataPreprocessor = DataPreprocessor()
if not dataPreprocessor.get_status():
    print("錯誤")
else:
    print(dataPreprocessor.get_video_and_annotation_paths())