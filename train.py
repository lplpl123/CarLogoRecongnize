import torch
import torchvision
import torch.nn as nn
from tools.preprocess import ReadData
from models.model import CNN
import torch.utils.data as Data


if __name__ == '__main__':
    # 数据预处理
    X_data, Y_labels = ReadData()

    for x in X_data:
        print(x[0].shape)

    train_loader = Data.DataLoader(
        dataset = X_data,
        batch_size = 50,
        shuffle = True
    )


    # torch.Size([])



    """
    (543, 3, 32, 32)
    (543,)
    34
    ['Kia 1', 'Opel', 'Mini', 'Dacia 1', 'Jeep', 'Honda 1', 
    'Tesla', 'Smart 2', 'Subaru', 'Daewoo 1', 'Land Rover', 
    'Acura', 'Lexus', 'Toyota', 'Mazda', 'GMC', 'Ford', 'Chevrolet 2', 
    'Porsche', 'Hyundai', 'BMW', 'Volkswagen', 'Alfa Romeo', 
    'Mitsubishi', 'Lancia 1', 'Nissan', 'Mercedes 1', 'Suzuki', 
    'Citroen', 'Skoda 1', 'Renault', 'Peugeot', 'Seat', 'Volvo 1']

    """

    # 超参数
    EPOCH = 50
    LR = 0.001

    # model
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):

        correct_train = 0
        total_train = 0

        for step, (batch_x, batch_y) in enumerate(train_loader):

            pred_y = cnn(batch_x)
            loss = loss_func(pred_y, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred_y.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

            if step % 50 == 0:
                train_accuracy = correct_train / total_train  # 计算训练准确率
                print("train_accuracy", train_accuracy)

                print('Epoch:', epoch, '| train loss: %.4f' % loss.data.numpy())




