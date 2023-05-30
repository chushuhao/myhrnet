import matplotlib.pyplot as plt
import re



def draw_val_loss(logfile):
    # 打开日志文件并逐行读取每个记录
    with open(logfile, 'r') as f:
        lines = f.readlines()

    # 用于匹配每个epoch的验证损失值的正则表达式模式
    val_loss_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} Loss: ([\d.]+)'

    # 存储每个epoch的验证损失值的列表
    val_losses = []

    for line in lines:
        # 检查该行是否包含验证损失值的字符串
        match = re.search(val_loss_pattern, line)
        if match:
            # 提取验证损失值并将其转换为浮点数
            val_loss = float(match.group(1))
            val_losses.append(val_loss)


    epochs = range(1, len(val_losses) + 1)
    # 绘制 val_loss 曲线
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    # 添加图例、标题和轴标签
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 显示图像
    plt.show()

def draw_train_loss(logfile):
    pattern = r'Epoch:\s\[(\d+)/(\d+)\]\sIter:\[(\d+)/(\d+)\],.*Loss:\s([\d.]+)'
    epoch_losses = {}
    with open(logfile, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                iteration = int(match.group(3))
                loss = float(match.group(5))
                epoch_losses[epoch] = loss

    epoch_losses = list(epoch_losses.values())
    epochs = range(1, len(epoch_losses) + 1)
    print(type(epoch_losses))
    # 绘制 val_loss 曲线
    plt.plot(epochs, epoch_losses, 'b', label='Train Loss')
    # 添加图例、标题和轴标签
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 显示图像
    plt.show()

def draw_val_and_train_loss(logfile):
    # 打开日志文件并逐行读取每个记录
    with open(logfile, 'r') as f:
        lines = f.readlines()

    # 用于匹配正则表达式模式
    train_loss_pattern = r'Epoch:\s\[(\d+)/(\d+)\]\sIter:\[(\d+)/(\d+)\],.*Loss:\s([\d.]+)'
    val_loss_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} Loss: ([\d.]+)'

    # 存储每个epoch的验证损失值的列表
    val_losses = []
    train_losses = {}

    for line in lines:
        val_match = re.search(val_loss_pattern, line)
        if val_match:
            # 提取验证损失值并将其转换为浮点数
            val_loss = float(val_match.group(1))
            val_losses.append(val_loss)
        train_match = re.search(train_loss_pattern, line)
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(5))
            train_losses[epoch] = loss

    epochs = range(1, len(train_losses) + 1)
    train_losses = list(train_losses.values())
    # 绘制 val_loss 曲线
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


logfile = '../output/logging/seg_hrnet_loveda_2023-05-06-07-13_train.log'
draw_val_and_train_loss(logfile)