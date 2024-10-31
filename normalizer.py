from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 평균 및 표준편차 계산 함수
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total_images_count = 0

    # 데이터셋 순회하며 평균과 표준편차 계산
    for images, _ in tqdm(loader):
        batch_images_count = images.size(0)  # 현재 배치 크기
        images = images.view(batch_images_count, images.size(1), -1)  # (B, C, H * W)

        # 각 배치의 평균 및 분산값을 누적
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_images_count

    # 전체 이미지 수로 나눠 평균과 표준편차 계산
    mean /= total_images_count
    std /= total_images_count
    return mean, std