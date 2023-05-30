import torch
from models import seg_hrnet
from torchviz import make_dot

def from_torchviz(model, channels, height, width):
    x = torch.randint(low=0, high=256, size=(1, channels, height, width), dtype=torch.uint8)  # 定义一个网络的输入值
    x = x.float().requires_grad_(True)
    y = model(x)  # 获取网络的预测值
    MyNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    MyNetVis.format = "png"
    # 指定文件生成的文件夹
    MyNetVis.directory = "../output"
    # 生成文件
    MyNetVis.view()


model = seg_hrnet.get_seg_model(num_class=7)
from_torchviz(model, channels=3, height=1024, width=1024)