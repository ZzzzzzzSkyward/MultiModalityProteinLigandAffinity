from header import *


def load_pretrained_weights(model, pretrained_file):
    """
    从预训练模型的文件.pth中加载权重并初始化模型

    参数：
        model: 待初始化的模型
        pretrained_file: 预训练模型的文件路径

    返回：
        初始化后的模型
    """
    # 加载预训练模型的权重
    pretrained_dict = torch.load(pretrained_file,
                                 map_location=torch.device('cpu'))

    # 获取模型的当前权重
    model_dict = model.state_dict()

    # 从预训练模型中筛选出与当前模型匹配的权重
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
    }

    # 更新模型的权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 返回初始化后的模型
    return model