"""
第三周作业：
设计一个以文本为输入的多分类任务，实验一下用LSTM模型的跑通训练。
如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 定义训练的关键字
KEY_WORD = "你"

# 获取常用的中文字
ALL_CHINESE_CHARS = [chr(code) for code in range(0x4e00, 0x9fa5)]

# --- 一些常用参数 ---
SAMPLE_SIZE = 4000  # 样本数量
MAX_LEN = 10  # 最长中文序列/最大类别
EMBEDDING_DIM = 32
HIDDEN_DIM = 16
LR = 1e-2
BATCH_SIZE = 40
EPOCHS = 20
TRAIN_RATIO = 0.8

# 固定随机种子
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def build_single_sample(size):
    """
    生成一个样本, 样本的生成方法，代表了我们要学习的规律
    随机生成一段中文序列，“你”字包含在第几位，就属于第几类
    :param size: 中文序列的长度
    :return: (中文序列, 类别-1（索引是从0开始的）)
    """
    # 获取size个不重复的汉字
    unique_chars = random.sample(ALL_CHINESE_CHARS, size)

    # 判断获取的汉字数组中是否存在“你”，若不存在则随机替换一个字符
    if KEY_WORD not in unique_chars:
        random_index = random.randint(0, size - 1)
        unique_chars[random_index] = KEY_WORD
    return "".join(unique_chars), unique_chars.index(KEY_WORD)


def build_batch_dataset(sample_num):
    """
    随机生成一批样本
    :param sample_num: 样本的数量
    :return: 样本值
    """
    data = []
    for i in range(sample_num):
        # 随机生成2 - MAX_LEN 字长度的中文序列
        data.append(build_single_sample(random.randint(2, MAX_LEN)))
    random.shuffle(data)
    return data


def build_vocab():
    """
    把获取的所有常用的中文字符构建成词表
    :return: 词表
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for index, post in enumerate(ALL_CHINESE_CHARS, start=2):
        vocab[post] = index
    return vocab


def encode(sent, vocab):
    """
    对每个中文序列进行填充和转译成向量
    :param sent: 中文序列样本
    :param vocab: 词表
    :return: 填充的中文序列向量
    """
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.Y = [lb for _, lb in data]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.Y[i], dtype=torch.long),
        )


class CustomModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        """
        初始化一个自定义的模型
        """
        super(CustomModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        # 使用双向的LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bias=True, batch_first=True, bidirectional=True)
        # 增加一个线性层作为输出层
        self.linear = nn.Linear(hidden_dim * 2, MAX_LEN)
        # 损失函数使用交叉熵损失函数来处理分类任务
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        输出预测值
        :param x: 输入向量
        :return: 返回预测结果
        """
        out, _ = self.lstm(self.embedding(x))
        return self.linear(out[:, -1, :])


def evaluate(model, loader):
    """
    通过传入的样本数据来对训练出来的模型进行准确率验证
    :param model: 训练后的模型
    :param loader: 测试样本数据
    :return: 正确率
    """
    model.eval()
    correct = wrong = 0
    with torch.no_grad():
        for X, Y in loader:
            y_forecast = model(X)
            for y_p, y_t in zip(y_forecast, Y):
                if y_p.argmax() == y_t:
                    correct += 1
                else:
                    wrong += 1
    return correct / (correct + wrong)


def main():
    # 随机生成一组训练集
    data = build_batch_dataset(SAMPLE_SIZE)
    # 把获取的所有常用的中文字符构建成词表
    vocab = build_vocab()
    print(f"获取的样本数：{len(data)}，词表大小：{len(vocab)}")

    # 把构建的数据，一部分做训练集，一部分做验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    # 把数据进行分批处理
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 建立模型
    model = CustomModel(num_embeddings=len(vocab))
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, Y in train_loader:
            y_forecast = model(X)
            loss = model.loss(y_forecast, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"轮次 {epoch:2d}/{EPOCHS}  误差={avg_loss:.4f}  准确率={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n------- 你可以输入你想验证的中文序列所属的类别 -------")
    model.eval()
    while True:
        str = input(f"\n请输入一串小于等于 '{MAX_LEN}' 个字符，且存在字符 “{KEY_WORD}” 的中文序列：")
        if KEY_WORD not in str:
            print(f"中文序列中需要包含字符 '{KEY_WORD}' ，请重新输入！")
            continue
        if len(str) > MAX_LEN:
            print(f"中文序列长度不能超过 '{MAX_LEN}' ，请重新输入！")
            continue
        with torch.no_grad():
            ids = torch.tensor([encode(str, vocab)], dtype=torch.long)
            y_forecast = model(ids)
            print(f"[第 {y_forecast.argmax().item() + 1} 类] {str}")


if __name__ == "__main__":
    main()
