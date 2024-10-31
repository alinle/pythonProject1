import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
1
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir=img_dir      # 이미지 경로
        self.transform = transform
        self.img_labels = []
        self.train = train

        if self.train:
            # os.listdir : img_dir 안에 있는 파일과 폴더 이름을 리스트로 반환
            # label 을 얻을 수 있음
            for class_dir in os.listdir(img_dir):
                class_dir_path = os.path.join(img_dir, class_dir)
                # 라벨 폴더 아래의 이미지들 주소 불러오기
                # 이미지 주소와 label 을 튜플로 저장하여 데이터셋 생성
                for img_file in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path,img_file)
                    self.img_labels.append((img_path, int(class_dir)))

        else:
            # test 파일 밑 경로를 가져오고 그 안의 이미지 저장
            # 이미지 이름에서 label 을 슬라이싱해서 배열에 저장하는 방식으로 데이터셋 생성
            self.img_labels = [(os.path.join(img_dir,img_file),int(img_file.split("_")[0])) for img_file in os.listdir(img_dir)]

    def __len__(self):
        return len(self.img_labels)

    # 반환 값 생성
    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Ensure label is a tensor
        label = torch.tensor(label)  # 레이블을 텐서로 변환
        return image, label