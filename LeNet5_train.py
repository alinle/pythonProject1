from LeNet5 import LeNet5
import os
import shutil
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import CustomDataset
from normalizer import calculate_mean_std
import matplotlib.pyplot as plt  # 그래프 생성을 위한 라이브러리 추가
#SDF
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    # 평균과 표춘편차를 계산하기 위한 transform
    transform_temp = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        ])

    temp_dataset = CustomDataset(img_dir="C:/Data Files/traffic_Data/DATA", transform=transform_temp, train=True)
    mean, std = calculate_mean_std(temp_dataset)
    print("Mean:", mean)
    print("Std:", std)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터 셋 생성
    train_dataset = CustomDataset(img_dir="C:/Data Files/traffic_Data/DATA", transform=transform, train=True)
    test_dataset = CustomDataset(img_dir="C:/Data Files/traffic_Data/TEST", transform=transform, train=False)

    # 데이터로더(데이터 셋을 batch단위로 나눔) 생성
    # shuffle: 데이터 무작위로 섞음
    # drop_last:  마지막 batch의 데이터 수가 batch_size 보다 작을때 마지막 뭉치를 버림
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = LeNet5().to(device)
    # SGD, 학습률 0.1
    sgd = SGD(model.parameters(), lr=1e-1)
    # 손실함수 -> crossentropy 사용
    loss_fn = CrossEntropyLoss()
    all_epoch = 50
    # 정확도 기록
    prev_acc = 0
    # 최대 정확도 기록
    highest_acc = 0.0
    model_directory = 'LeNet_models'  # 모델이 저장될 디렉토리
    accuracy_list=[]

    for current_epoch in range(all_epoch):
        model.train() # 모델 train 활성
        # 모델 훈련 과정
        for idx, (train_x, train_label) in enumerate(train_loader):
            # loder 에 배치단위로 저장된 훈련데이터를 GPU에 전송
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            # sgd 는 기울기를 누적해서 초기화 해줘야함
            sgd.zero_grad()
            # 모델에 입력을 전달한 후 예측값 계산
            predict_y = model(train_x)
            # 실제 라벨과 비교하여 손실값 계산
            loss = loss_fn(predict_y, train_label)
            loss.backward()
            sgd.step()

        # 모델 test 모드로 변경
        model.eval()
        all_correct_num = 0 # 예측 성공 횟수 저장
        all_sample_num = 0  # 총 예측 횟수 저장
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()  # 기울기 계산에서 분리시켴 계산 횟수를 줄임
            # 예측 결과 차원 조정
            # 가장 높은 확률을 가진 클래스 인덱스 얻기
            predict_y = torch.argmax(predict_y, dim=1)

            # 예측과 실제 레이블 비교
            current_correct_num = (predict_y == test_label).sum()  # 정확한 예측 수 계산
            all_correct_num += current_correct_num.item()  # 전체 정확한 예측 수에 추가
            all_sample_num += test_label.size(0)  # 전체 샘플 수 증가

        acc = all_correct_num / all_sample_num
        accuracy_list.append(acc)  # 정확도 리스트에 추가
        print(f'epoch {current_epoch + 1} \n' + 'accuracy: {:.3f}'.format(acc), flush=True)

        # 최고 정확도 갱신 및 모델 저장
        if acc > highest_acc:  # 현재 정확도가 최고 정확도보다 높으면
            highest_acc = acc  # 최고 정확도 갱신

            # 이전 모델 파일 삭제

            for file in os.listdir(model_directory):
                file_path = os.path.join(model_directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')

            # 새로운 모델 저장
            torch.save(model, os.path.join(model_directory, 'LeNet_{:.3f}.pkl'.format(acc)))
            print(f'Model saved with accuracy: {acc:.3f}', flush=True)

        prev_acc = acc

    print("Training finished")
