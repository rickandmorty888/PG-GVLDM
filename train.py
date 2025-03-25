import os
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import wcatt as at
from transformers import GPT2Tokenizer
import h5py
import csv
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

Main_ROIs=['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', 'V3A', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT',
           'MT', 'PCV', 'STV', 'PH', 'DVT', 'V6A', 'VMV1', 'VMV3', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', 'VVC']
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/share/home/huafuchen01/huangwei/DiweiWu/GPT2/decodeGPT/add_pad/", type=str, help='')
    parser.add_argument('--save_model_path', default="save_models_wcatt", type=str, help='')
    parser.add_argument('--train_raw_path', default='/share/home/huafuchen01/huangwei/DiweiWu/simpleData.txt', type=str, help='')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=5000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1.5e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=1, type=int, required=False, help='print log steps')
    parser.add_argument('--save_epoch', default=1, type=int, required=False, help='save epochs')
    return parser.parse_args()

def load_model(model_path):
    model = at.decodeGPTLMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return model, tokenizer

def train_data_loader(args, train_data_path, tokenizer, shuffle):
    text_list = []
    subj_list = []
    stms_list = []
    lastsubj = 'subj00'#初始化上一个被试编号
    with open(train_data_path, 'rb') as f:
        data = f.read().decode("utf-8")
        train_data = data.split("\n")
        train_data.pop()#删除最后空行
        print("数据总行数:{}".format(len(train_data)))
        # read task and label
        load_num = 0
        for txt in tqdm(train_data):
            text_split = txt.split("#", 1)
            stm, text = text_split
            subj='subj' + text[8:10]
            subj_path = '/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step02_readData/' + subj + '/beta-hdf5/HCP_MMP1/'
            subj_stm_path = '/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step00_excel/trailID-' + subj + '.csv'


            allROIs = list()
            datalist=os.listdir(subj_path)
            datalist.sort()
            for file in datalist:  # 获取所有ROI名字
                allROIs.append(file)

            #如果被试编号变化，重新定位数据集位置
            if( subj != lastsubj ):
                #打开刺激列表
                stm_data = []
                with open(subj_stm_path) as csvfile:
                    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                    for row in csv_reader:  # 将csv 文件中的数据保存到data中
                        stm_data.append(row[11])  # 选择某一列加入到data数组中
                stm_data.pop(0) #去除列标签
            lastsubj = subj#更新上一个被试编号


            if ( stm_data[int(stm)] == 'False'):
                load_num = load_num + 1
                text_ids = tokenizer.encode('<|endoftext|>' + text + '<|endoftext|>')
                text_list.append(text_ids)
                # read rois data
                subj_list.append(subj)
                stms_list.append(int(stm))

    print(str(load_num)+' train samples loaded')
    dataset = MyDataset(text_list, subj_list, stms_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader

class MyDataset(Dataset):
    def __init__(self, text_list, subj_list, stms_list):
        self.text_list = text_list
        self.subj_list = subj_list
        self.stms_list = stms_list


    def __getitem__(self, index):
        text_ids = self.text_list[index]
        subj_id = self.subj_list[index]
        fmri_stm = self.stms_list[index]

        return text_ids, subj_id, fmri_stm

    def __len__(self):
        return len(self.text_list)


def collate_fn(batch):
    text_len_list = []
    for btc_idx in range(len(batch)):
        text_len = len(batch[btc_idx][0])
        text_len_list.append(text_len)
    max_text_len = max(text_len_list)
    data = []
    for btc_idx in range(len(batch)):
        text_len = len(batch[btc_idx][0])
        data.append(list(batch[btc_idx]))
        # use 'padding' to pad
        data[btc_idx][0].extend([50257] * (max_text_len - text_len))
        data[btc_idx][0] = torch.tensor(data[btc_idx][0], dtype=torch.long)
    return data

def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=50257, reduction='sum')
    pre = shift_logits.view(-1, shift_logits.size(-1))
    gt = shift_labels.view(-1)
    loss = loss_fct(pre, gt)

    _, preds = shift_logits.max(dim=-1)

    not_ignore = shift_labels.ne(50257)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy



def train(args, model, tokenizer, dataloader):
    num_training_steps = args.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    batch_steps = 0
    cur_epochs = 0

    loss_list=[]
    accuracy_list=[]
    begin = time.perf_counter()

    # 读取4个被试roi数据到内存
    main_roi_datasets = {}
    other_roi_datasets = {}
    main_data = {}
    other_data = {}
    subjs = ['subj01', 'subj02', 'subj05', 'subj07']
    for subj in subjs:
        subj_path = '/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step02_readData/' + subj + '/beta-hdf5/HCP_MMP1/'
        allROIs = []
        mainROIs = []
        otherROIs = []
        datalist = os.listdir(subj_path)
        datalist.sort()
        for file in datalist:  # 获取所有ROI名字
            allROIs.append(file)
        for file in allROIs:  # 获取主要和次要ROI文件名字
            ism = 0
            for vr in Main_ROIs:
                if (file.startswith(vr + '-')):
                    mainROIs.append(file)
                    ism = 1
            if ism == 0:
                otherROIs.append(file)
        # 读取主要ROI数据
        roi_datasets = []
        for roi_path in mainROIs:
            hdfFile = h5py.File(subj_path + roi_path, 'r')
            roi_dataset = hdfFile.get('beta')
            roi_datasets.append(roi_dataset)
        main_roi_datasets[subj] = roi_datasets
        # 读取次要ROI数据
        roi_datasets = []
        for roi_path in otherROIs:
            hdfFile = h5py.File(subj_path + roi_path, 'r')
            roi_dataset = hdfFile.get('beta')
            roi_datasets.append(roi_dataset)
        other_roi_datasets[subj] = roi_datasets

    print('start trans dim')
    # 为特征增加ROI编号维度，然后拼接
    for subj in subjs:
        m_embeds = []
        o_embeds = []
        for stm in range(0, 100):
            m_embed = np.array([])
            o_embed = np.array([])
            for r in range(0, 29):
                    m_roi = np.concatenate((np.array([r]), main_roi_datasets[subj][r][stm, :]), axis=0)
                    m_embed = np.concatenate((m_embed, m_roi), axis=0)
            for r in range(0, 151):
                    o_roi = np.concatenate((np.array([r]), other_roi_datasets[subj][r][stm, :]), axis=0)
                    o_embed = np.concatenate((o_embed, o_roi), axis=0)
            m_embeds.append(m_embed)
            o_embeds.append(o_embed)
        main_data[subj] = np.array(m_embeds)
        other_data[subj] = np.array(o_embeds)
    # 获取最大维度，padding, reshape
    max_mdim = max(main_data['subj01'][0,:].size, main_data['subj02'][0,:].size, main_data['subj05'][0,:].size, main_data['subj07'][0,:].size)
    max_odim = max(other_data['subj01'][0,:].size, other_data['subj02'][0,:].size, other_data['subj05'][0,:].size, other_data['subj07'][0,:].size)
    mrow = (max_mdim // 768) + 1
    orow = (max_odim // 768) + 1
    for subj in subjs:
        main_data[subj] = np.pad(main_data[subj],  ((0,0), (0,mrow*768-main_data[subj].shape[1])))
        other_data[subj] = np.pad(other_data[subj], ((0,0), (0,orow*768-other_data[subj].shape[1])))
        main_data[subj].shape=(100, mrow, 768)
        other_data[subj].shape = (100, orow, 768)
        main_data[subj] = torch.FloatTensor(main_data[subj])
        other_data[subj] = torch.FloatTensor(other_data[subj])
    print('finish trans dim')

    for epoch in range(args.epochs):
        cur_epochs += 1
        for batch in dataloader:
            batch_steps += 1
            text_ids = torch.LongTensor(len(batch), len(batch[0][0]))
            main_fmri_embeds = torch.FloatTensor(len(batch), 26, 768)
            other_fmri_embeds = torch.FloatTensor(len(batch), 113, 768)
            for btc_idx in range(len(batch)):
                text_ids[btc_idx, :] = batch[btc_idx][0]
                subj_id = batch[btc_idx][1]
                stm = batch[btc_idx][2]
                # 根据subj与stm索引读取fmri信号
                main_fmri_embeds[btc_idx, :, :]=main_data[subj_id][stm, :, :]
                other_fmri_embeds[btc_idx, :, :] = other_data[subj_id][stm, :, :]
            inputs = {"input_ids": text_ids.to(device),
                      "main_fmri_embeds": main_fmri_embeds.to(device),
                      "other_fmri_embeds": other_fmri_embeds.to(device)}


            outputs = model(**inputs, labels=text_ids.to(device))

            # loss = outputs.loss
            loss, acc= calculate_loss_and_accuracy(outputs,text_ids.to(device), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


            if batch_steps % args.log_step == 0:
                end = time.perf_counter()
                cost_time = end - begin
                begin = time.perf_counter()
                loss_list.append(loss.cpu().detach().numpy())
                accuracy_list.append(acc.cpu().detach().numpy())
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, timecost {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc, cost_time))

        if cur_epochs % args.save_epoch == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.save_model_path + '/epoch' + str(cur_epochs), safe_serialization = False)
            tokenizer.save_pretrained(args.save_model_path + '/epoch' + str(cur_epochs))
            log_path = '/share/home/huafuchen01/huangwei/DiweiWu/GPT2/decodeGPT/save_models_wcatt/epoch' + str(cur_epochs) + '.h5'
            f = h5py.File(log_path, "w")
            f.create_dataset("loss_list", data=loss_list)
            f.create_dataset("acc_list", data=accuracy_list)
            f.close()
            loss_list=[]
            accuracy_list=[]



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    model, tokenizer = load_model(args.model_path)
    train_dataloader = train_data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=True)
    train(args, model, tokenizer, train_dataloader)
    # model, tokenizer = load_model(args.save_model_path)
    # eval_dataloader = train_data_loader(args, args.train_raw_path, Main_ROIs, tokenizer=tokenizer, shuffle=True)
    # evaluate(args, model, eval_dataloader)



