import json
import os
import random
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F


label_mapping={
'label0':0, 'label1':1, 'label2':2, 'label3':3
}



def getPool(dataPath,num_labels):
    # 创建一个空列表作为池
    pool=[]
    # 根据标签数量循环创建子列表
    for i in range(num_labels):
        pool.append([])

    # 打开文件并读取每一行
    with open(dataPath, 'r',encoding="utf-8") as file:
        for line in file:
            # 解析每一行JSON数据
            js=json.loads(line)
            # 根据标签映射关系，将文本添加到对应的子列表中
            pool[label_mapping[js['intent']]].append(str(js['text']))

    return pool


def train(model,device,pool,iter,optimizer,tokenizer):
    # 生成一个数字列表
    numbers = []
    for i in range(len(pool)):
        numbers.append(i)
    # 使用tqdm库显示迭代进度
    for i in tqdm(range(0,iter)):

        # 从数字列表中随机选择两个数字
        selected_numbers = random.sample(numbers, 2)
        # 从pool中选择两个正样本
        positive = random.sample(pool[selected_numbers[0]], 2)
        # 从pool中选择一个负样本
        negative = random.choice(pool[selected_numbers[1]])

        # 将正样本和负样本赋值给text_a, text_b, text_c
        text_a = positive[0]
        text_b = positive[1]
        text_c = negative


        # 对text_a, text_b, text_c进行分词和编码，得到inputs_a, inputs_b, inputs_c
        inputs_a = tokenizer(text_a, return_tensors='pt', padding=True, truncation=True)
        inputs_b = tokenizer(text_b, return_tensors='pt', padding=True, truncation=True)
        inputs_c = tokenizer(text_c, return_tensors='pt', padding=True, truncation=True)

        # 将inputs_a, inputs_b, inputs_c转移到指定的设备上
        inputs_a.to(device)
        inputs_b.to(device)
        inputs_c.to(device)

        # 将inputs_a, inputs_b, inputs_c输入到模型中进行前向传播，得到outputs_a, outputs_b, outputs_c
        outputs_a = model(**inputs_a)
        outputs_b = model(**inputs_b)
        outputs_c = model(**inputs_c)


        # 从outputs_a, outputs_b, outputs_c中提取最后一个隐藏层的第一个时间步的嵌入向量
        embed_a = outputs_a.last_hidden_state[:, 0, :]
        embed_b = outputs_b.last_hidden_state[:, 0, :]
        embed_c = outputs_c.last_hidden_state[:, 0, :]

        # 计算正样本的损失和负样本的损失
        positive_loss = F.relu(1-F.cosine_similarity(embed_a, embed_b))
        negative_loss = F.relu(F.cosine_similarity(embed_a, embed_c))
        # 将正样本损失和负样本损失相加得到对比损失
        contrastive_loss = positive_loss + negative_loss

        # 打印对比损失
        print(f'contrastive_loss:{contrastive_loss}')

        # 清空梯度
        optimizer.zero_grad()
        # 对对比损失进行反向传播
        contrastive_loss.backward()
        # 更新模型参数
        optimizer.step()


if __name__ == '__main__':
    iter=100000
    pool=getPool('./data/mix.jsonl', 4)

    model_name='./bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device="cpu"
    if torch.cuda.is_available():
        model.cuda()
        device="cuda"
    else:
        device="cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train(model,device,pool,iter,optimizer,tokenizer)




