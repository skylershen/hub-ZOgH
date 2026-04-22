"""
第二周作业：
基于pytorch框架编写模型训练
尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 定义一个全局常量，表示向量的大小
VECTOR_SIZE = 10


class CustomModel(nn.Module):
    def __init__(self):
        """
        初始化一个自定义的模型
        """
        super(CustomModel, self).__init__()

        # 增加一个线性层，对输入数据做线性变换：y=Ax+b，输入样本大小为VECTOR_SIZE，输出样本大小为 VECTOR_SIZE
        self.linear = nn.Linear(VECTOR_SIZE, VECTOR_SIZE)

        # 损失函数使用交叉熵损失函数来处理分类任务
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        输出预测值
        :param x: 输入向量
        :return: 返回预测结果
        """
        return self.linear(x)


def build_single_sample():
    """
    生成一个样本, 样本的生成方法，代表了我们要学习的规律
    随机生成一个 VECTOR_SIZE 维向量，哪一维数字最大就属于第几类
    :return: (随机向量, 类别)
    """
    x = np.random.random(VECTOR_SIZE)
    return x, x.argmax()


def build_batch_dataset(sample_num):
    """
    随机生成一批样本
    :param sample_num: 样本的数量
    :return: 样本值
    """
    x_val = []
    y_val = []
    for i in range(sample_num):
        x, y = build_single_sample()
        x_val.append(x)
        y_val.append(y)
    return torch.FloatTensor(np.array(x_val)), torch.LongTensor(y_val)


def evaluate(model, test_sample_num):
    """
    随机生成多组样本数据，来对训练出来的模型进行准确率验证
    :param model: 训练后的模型
    :param test_sample_num: 测试样本数量
    :return: 正确率
    """
    model.eval()
    x, y = build_batch_dataset(test_sample_num)
    unique_vals, counts = torch.unique(y, return_counts=True)
    keys = [f"第 {i + 1} 类" for i in unique_vals.tolist()]
    print(f"当前为'{VECTOR_SIZE}'维向量，本次预测集中各类别数量: {dict(zip(keys, counts.tolist()))}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_forecast = model(x)
        for y_p, y_t in zip(y_forecast, y):
            if y_p.argmax() == y_t:
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%.4f" % (correct, accuracy))
    return accuracy


def main():
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 7000  # 每轮训练总共训练的样本总数
    learning_rate = 0.01  # 学习率
    log = []

    # 建立模型
    model = CustomModel()
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 随机生成一组训练集
    train_x, train_y = build_batch_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            # 前向传播，输出预测值
            y_forecast = model(x)

            # 计算损失，输出损失值
            loss = model.loss(y_forecast, y)

            # 反向传播：计算梯度 + 梯度清零
            optim.zero_grad()
            loss.backward()

            # 更新参数
            optim.step()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, 100)
        log.append([acc, float(avg_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
