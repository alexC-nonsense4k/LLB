import os
import random
import time
import datetime
from transformers import BertTokenizer,BertForSequenceClassification,AdamW,BertConfig,AutoTokenizer,get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset,random_split,DataLoader,RandomSampler
import json
import numpy as np
import time
import datetime



label_mapping={
'label0':0, 'label1':1, 'label2':2, 'label3':3
}



def loadJsonl(jsonlPath):
    # 创建一个空列表，用于存储解析后的 JSON 数据
    json_list=[]
    # 打开指定路径下的 JSONL 文件，并读取其中的内容
    with open('./data/mix.jsonl', 'r',encoding="utf-8") as file:
        # 逐行读取文件中的内容
        for line in file:
            # 使用 json.loads() 方法将每行内容解析为 JSON 对象，并添加到 json_list 列表中
            json_list.append(json.loads(line))
    # 返回解析后的 JSON 列表
    return json_list



def getTrainData(json_list):
    # 存储句子的列表
    sentences=[]
    # 存储标签的列表
    labels=[]
    # 遍历json_list列表
    for js in json_list:
        # 将句子添加到sentences列表中
        sentences.append(str(js['text']))
        # 将标签添加到labels列表中，根据label_mapping进行映射
        labels.append(label_mapping[js['intent']])
    # 将句子和标签组合成元组列表
    combined = list(zip(sentences, labels))
    # 对元组列表进行随机打乱
    random.shuffle(combined)
    # 解包元组列表，分别得到打乱后的句子和标签列表
    shuffled_sentences, shuffled_labels = zip(*combined)
    # 将打乱后的句子列表转换为列表类型
    sentences = list(shuffled_sentences)
    # 将打乱后的标签列表转换为列表类型
    labels = list(shuffled_labels)
    # 返回打乱后的句子列表和标签列表
    return sentences,labels



def getDataset(tokenizer, sentences, labels):
    # 初始化最大长度变量
    max_len = 0
    # 遍历句子列表，计算最大长度
    for sen in tqdm(sentences):
        input_ids = tokenizer.encode(sen, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    # 初始化输入ID列表和注意力掩码列表
    input_ids = []
    attention_masks = []

    # 遍历句子列表，对每个句子进行编码并添加到对应列表中
    for sen in tqdm(sentences):
        encoded_dict = tokenizer.encode_plus(
            sen,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # 将输入ID列表和注意力掩码列表转换为Tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # 创建TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # 计算训练集和验证集的大小
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    # 将数据集划分为训练集和验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建训练集和验证集的数据加载器
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)

    # 返回训练集和验证集的数据加载器
    return train_dataloader, validation_dataloader


def flat_accuracy(preds, labels):
    # 将预测结果转换为扁平化的一维数组
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 将标签转换为扁平化的一维数组
    labels_flat = labels.flatten()
    # 计算预测结果和标签相等的数量，并除以标签的数量，得到准确率
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    # 将传入的elapsed参数四舍五入为整数
    elapsed_rounded = int(round((elapsed)))
    # 将整数转换为时间差对象
    return str(datetime.timedelta(seconds=elapsed_rounded))




def train(train_dataloader,validation_dataloader,batch_size,epochs,optimizer,scheduler,model,device):
    """
    训练模型函数

    Args:
        train_dataloader (DataLoader): 训练集数据加载器
        validation_dataloader (DataLoader): 验证集数据加载器
        batch_size (int): 批处理大小
        epochs (int): 训练周期数
        optimizer (Optimizer): 优化器
        scheduler (LRScheduler): 学习率调度器
        model (nn.Module): 待训练的模型
        device (str): 设备类型，'cpu'或'cuda'

    Returns:
        None
    """
    # 设置随机种子以确保实验可复现
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 初始化训练统计信息列表
    training_stats = []
    # 记录训练总时长
    total_t0 = time.time()

    # 训练循环
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # 记录当前训练周期的开始时间
        t0 = time.time()
        # 初始化训练损失
        total_train_loss = 0
        # 设置模型为训练模式
        model.train()

        # 遍历训练数据加载器
        for step, batch in enumerate(train_dataloader):
            # 每隔40步打印一次训练进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 将输入数据、掩码和标签移动到指定设备上
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # 清零梯度
            model.zero_grad()

            # 前向传播
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

            # 获取损失和logits
            loss = result.loss
            logits = result.logits

            # 累加训练损失
            total_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 更新学习率调度器
            scheduler.step()

        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataloader)
        # 计算训练周期时长
        training_time = format_time(time.time() - t0)

        # 打印训练周期信息
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # 开始验证
        print("")
        print("Running Validation...")
        # 记录当前验证周期的开始时间
        t0 = time.time()
        # 设置模型为评估模式
        model.eval()
        # 初始化验证准确率和损失
        total_eval_accuracy = 0
        total_eval_loss = 0
        # 验证步数计数器
        nb_eval_steps = 0

        # 遍历验证数据加载器
        for batch in validation_dataloader:
            # 将输入数据、掩码和标签移动到指定设备上
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # 不计算梯度
            with torch.no_grad():
                # 前向传播
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

                # 获取损失和logits
                loss = result.loss
                logits = result.logits

                # 累加验证损失
                total_eval_loss += loss.item()

                # 计算准确率
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 计算平均验证准确率和损失
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        # 计算验证时长
        validation_time = format_time(time.time() - t0)

        # 打印验证周期信息
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # 将训练周期信息添加到训练统计信息列表中
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


if __name__ == '__main__':
    model_name='./bert-base-chinese'
    output_dir = './finetune_output'
    epochs=1
    batch_size=22
    json_list=loadJsonl('./data/mix.jsonl')
    sentences,labels=getTrainData(json_list)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    train_dataloader,validation_dataloader=getDataset(tokenizer,sentences,labels)
    model=BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False
    )
    device="cpu"
    if torch.cuda.is_available():
        model.cuda()
        device="cuda"
    else:
        device="cpu"
    optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-9)
    total_step=len(train_dataloader)*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_step)
    train(train_dataloader,validation_dataloader,batch_size,epochs,optimizer,scheduler,model,device)