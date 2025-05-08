from common import BaseExperiment
from model import CLPRNet as Model
from dataset import MyDataset, CHARACTER
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import provinces1, provinces2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def IoU_multi(pred_boxes, target_boxes, eps=1e-6):
    """
    pred_boxes: shape[N, 4]
    target_boxes: shape[N, 4]
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = torch.split(pred_boxes, 1, dim=-1)
    target_x1, target_y1, target_x2, target_y2 = torch.split(target_boxes, 1, dim=-1)

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    iou = inter_area / (union_area + eps)

    return iou

def IOU(box,other_boxes):
    box_area = (box[2]-box[0])*(box[3]-box[1])
    other_boxes_area = (other_boxes[:,2]-other_boxes[:,0]) * (other_boxes[:,3]-other_boxes[:,1])
    x1 = torch.max(box[0],other_boxes[:,0])
    y1 = torch.max(box[1],other_boxes[:,1])
    x2 = torch.min(box[2],other_boxes[:,2])
    y2 = torch.min(box[3],other_boxes[:,3])
    Min = torch.zeros(1, device=box.device)
    w,h = torch.max(Min,x2-x1),torch.max(Min,y2-y1)
    overlap_area = w*h
    iou = overlap_area / (box_area+other_boxes_area-overlap_area+1e-6)
    return iou

def NMS(boxes, C = 0.5):
    #boxes：[c, x1, y1, x2, y2, other]
    if len(boxes) == 0:
        return []
    sort_boxes = boxes[boxes[:,0].argsort(descending=True)]
    keep = []
    while len(sort_boxes)>0:
        ref_box = sort_boxes[0]
        keep.append(ref_box)
        if len(sort_boxes) > 1:
            other_boxes = sort_boxes[1:]
            sort_boxes = other_boxes[torch.where(IOU(ref_box[1:5], other_boxes[:,1:5])<C)] 
        else:
            break
    return torch.stack(keep)
  
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, predict, target):
        pt = predict
        loss = - ((1 - self.alpha) * ((1 - pt+1e-5) ** self.gamma) * (target * torch.log(pt+1e-5)) +  self.alpha * (
                (pt+1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt+1e-5)))
 
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class BCEWithWeightLoss(torch.nn.Module):
    def __init__(self, weight=[1,1], reduction='mean'):
        super(BCEWithWeightLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
 
    def forward(self, inputs, target):
        loss = -(self.weight[1]*target*torch.log(inputs+1e-7) + self.weight[0]*(1-target)*torch.log(1-inputs+1e-7))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha:torch.Tensor, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        alpha = alpha.to(pred.device)
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss 
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

class Experiment(BaseExperiment):

    def __init__(self, **parameter) -> None:
        super().__init__(**parameter)
        self.LR_STEP_SIZE = parameter['LR_STEP_SIZE']
        self.LR_STEP_GAMMA = parameter['LR_STEP_GAMMA']
        self.MOMENTUM = parameter['MOMENTUM']
        self.WEIGHT_DECAY = parameter['WEIGHT_DECAY']
        self.BETAS = parameter['BETAS']
        self.file_name = ''
        self.x_mask = mask[:,:,0].to(self.DEVICE).unsqueeze_(dim=2)
        self.y_mask = mask[:,:,1].to(self.DEVICE).unsqueeze_(dim=2)
        self.alpha = [3]*31 + [1]*24 + [1]*10 + [5]*7 + [0.1]
        self.alpha = torch.tensor(self.alpha,device=self.DEVICE)
        
    def load_model(self, pretrain:str=None):
        self.model = Model()   
        self.model = self.model.to(self.DEVICE) 
        if pretrain:
            self.print(f"pretrain: {pretrain}")
            self.model.load_state_dict(torch.load(os.path.join(self.WORKSPACE,pretrain)))
        if self.CHECKPOINT:
            self.model.load_state_dict(torch.load(os.path.join(self.WORKSPACE,self.CHECKPOINT))['parameter'])
            self.START_EPOCH = torch.load(os.path.join(self.WORKSPACE,self.CHECKPOINT))['epoch'] + 1
                
    def load_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR, momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
        if self.CHECKPOINT:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.WORKSPACE,self.CHECKPOINT))['optimizer'])

    def load_scheduler(self):
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[80,85,90], gamma=self.LR_STEP_GAMMA)
        if self.CHECKPOINT:
            state_dict = torch.load(os.path.join(self.WORKSPACE,self.CHECKPOINT))['scheduler']
            self.scheduler.load_state_dict(state_dict)

    def forward(self, data):
        x, _, _, _, _, _, _, _ = data 
        x = x.to(self.DEVICE)
        pred = self.model(x)  
        return pred

    def loss(self, data, pred):     #regression    regression -> [b, 4, 4, (x,y,w,h,c)]
        img, lp_at, ch_at, bboxs, lp, lurds, lp_at_rec_facal, _ = data
        y_detection, y_recognition, pred_at_lp, pred_at_ch = pred

         # todo  at loss
        lp_at = lp_at.to(self.DEVICE).unsqueeze(dim=1)
        ch_at = ch_at.to(self.DEVICE)
        loss_at = BCEWithWeightLoss(weight=[0.1, 0.9])(pred_at_lp, lp_at) + BCEWithWeightLoss(weight=[0.05, 0.95])(pred_at_ch, ch_at)

        # todo  location loss
        bboxs = bboxs.to(self.DEVICE)
        lurds = lurds.to(self.DEVICE)  
        obj = torch.sum(bboxs, dim=3)>0
        noobj = torch.sum(bboxs, dim=3)==0

        l, t, r, b = torch.split(y_detection[:,:,:,:4], 1, dim=-1)
        l = self.x_mask - l*img.shape[3]  
        t = self.y_mask - t*img.shape[2]
        r = self.x_mask + r*img.shape[3]
        b = self.y_mask + b*img.shape[2]
        iou = IoU_multi(torch.flatten(lurds, start_dim=0, end_dim=2),torch.flatten(torch.concat([l, t, r, b], dim=3), start_dim=0, end_dim=2))
        iou = iou.view(bboxs.shape[:3])

        loss_location = -torch.log(iou + 1e-6) * obj
        loss_location = torch.sum(loss_location)/(torch.sum(obj)+ 1e-6)

        # todo  confidence loss
        confidence_location = iou.detach().float()
        loss_confidence_location = nn.MSELoss(reduction='none')(y_detection[:,:,:,4], confidence_location) * obj + \
                            0.1*nn.MSELoss(reduction='none')(y_detection[:,:,:,4], torch.zeros_like(confidence_location, device=self.DEVICE)) * noobj   
        loss_confidence_location = torch.mean(loss_confidence_location)  

        lp = lp.to(self.DEVICE)
        lp_obj = torch.sum(lp, dim=3)>0
        lp_at_rec_facal = lp_at_rec_facal.to(self.DEVICE)

        loss_classify = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(torch.flatten(y_recognition[:,:,:,0:73], start_dim=0, end_dim=2), torch.flatten(lp[:,:,:,0], start_dim=0, end_dim=2))      
        for j in range(1,8):
            loss_classify += nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(torch.flatten(y_recognition[:,:,:,73*j:73*(j+1)], start_dim=0, end_dim=2), torch.flatten(lp[:,:,:,j], start_dim=0, end_dim=2))

        loss_classify = torch.sum(loss_classify /8 * torch.flatten(lp_obj, start_dim=0, end_dim=2))/(torch.sum(lp_obj)+ 1e-6)
    
        loss = 0.2*loss_location + loss_confidence_location + 0.2*loss_classify + 10*loss_at
        return loss
    
    def before_val(self):
        self.count_iou = 0
        self.iou_list = [torch.zeros(1, device=self.DEVICE)]
        self.sample_num = 0
        self.pred_num = 1e-5
        self.count_lp = 0
        self.count_iou_lp = 0
    
    def evaluate(self, data, pred):
        img, _, _, _, _, _, _, lp_lurd  = data
        y_detection, y_recognition, _, _ = pred
        y_recognition = y_recognition.repeat_interleave(int(y_detection.shape[1]/y_recognition.shape[1]), dim=1)
        y_recognition = y_recognition.repeat_interleave(int(y_detection.shape[2]/y_recognition.shape[2]), dim=2)
        for index in range(y_detection.shape[0]):
            
            lp_lurd_list = lp_lurd[index].split(';')
            lp_list = []
            lurd_list = []
            for i in lp_lurd_list:
                lurd, lp = i.split('-')
                lp_list.append(np.array(lp.split(',')).astype('int32'))
                lurd_list.append(np.array(lurd.split(',')).astype('int32'))

            l, t, r, b, c = torch.split(y_detection[index,:,:,:5], 1, dim=-1)
            l = self.x_mask - l*img.shape[3]  
            t = self.y_mask - t*img.shape[2]
            r = self.x_mask + r*img.shape[3]
            b = self.y_mask + b*img.shape[2]
            ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8 = torch.split(y_recognition[index,:,:,:], 73, dim=-1)
            ch1 = F.softmax(ch1, dim=2)
            ch2 = F.softmax(ch2, dim=2)
            ch3 = F.softmax(ch3, dim=2)
            ch4 = F.softmax(ch4, dim=2)
            ch5 = F.softmax(ch5, dim=2)
            ch6 = F.softmax(ch6, dim=2)
            ch7 = F.softmax(ch7, dim=2)
            ch8 = F.softmax(ch8, dim=2)
            ch = torch.min(torch.stack([torch.max(ch1, dim=2)[0], torch.max(ch2, dim=2)[0], torch.max(ch3, dim=2)[0], torch.max(ch4, dim=2)[0], torch.max(ch5, dim=2)[0], torch.max(ch6, dim=2)[0], torch.max(ch7, dim=2)[0], torch.max(ch8, dim=2)[0]],dim=2), dim=2)[0]
            c = c.squeeze_(dim=2) * ch

            out = torch.flatten(torch.concat([c.unsqueeze_(dim=2), l, t, r, b, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8], dim=2), start_dim=0, end_dim=1)
            out = out[torch.where(out[:,0]>0.3)]
            out = NMS(out, 0.3)

            self.sample_num += len(lp_list)
            self.pred_num += len(out)

            for i in range(len(lp_list)):
                for j in range(len(out)):
                    iou = self.iou(torch.from_numpy(lurd_list[i]).to(self.DEVICE), torch.stack([out[j][1], out[j][2], out[j][3], out[j][4]]))
                    self.iou_list.append(iou)
                    if  iou > 0.7:
                        self.count_iou += 1

                    lp_pred = []
                    for k in range(8):
                        lp_pred.append(torch.argmax(out[j][5+k*73:5+(k+1)*73]))
                    lp_pred = torch.stack(lp_pred)
                    if (lp_pred==torch.from_numpy(lp_list[i]).to(self.DEVICE)).all():
                        self.count_lp += 1

                    if (lp_pred==torch.from_numpy(lp_list[i]).to(self.DEVICE)).all() and iou > 0.6:
                        self.count_iou_lp += 1      

    def after_val(self):
        print(f"Val IoU Detection Accuracy:{self.count_iou/len(self.val_dataset):>7f}")
        self.iou_list = torch.concat(self.iou_list,dim=0)
        print(f"Ave IoU:{torch.mean(self.iou_list):>7f}")
        print(f"Val Recognition Accuracy:{self.count_lp/len(self.val_dataset):>7f}")
        print(f"Val Recognition and Detection Accuracy:{self.count_iou_lp/len(self.val_dataset):>7f}")
        print(f"Val sample_num:{self.sample_num:>7f}")
        print(f"Val pred_num:{self.pred_num:>7f}")
        print(f"Val recall:{self.count_iou_lp/self.sample_num:>7f}")
        print(f"Val precision:{self.count_iou_lp/self.pred_num:>7f}")
        if self.mode == 'train':
            with open(os.path.join(self.OUTPUT, self.TIMESTAMP + "-_Accuracy.csv"),'a') as f:
                f.write(str(self.STEP) + ',' + str(self.count_iou/len(self.val_dataset)) + ',' + str(self.count_lp/len(self.val_dataset))+ ',' + str(self.count_iou_lp/len(self.val_dataset)))
                f.write('\n')
            with open(os.path.join(self.OUTPUT, self.TIMESTAMP + "-_Accuracy.csv"),'r',encoding='utf8') as f:
                txts = f.readlines()
                txts = [i.rstrip("\n") for i in txts]
                Accuracy = np.array([i.split(',') for i in txts]).astype(np.float32)
            plt.figure('Accuracy')
            plt.plot(Accuracy[:,0], Accuracy[:,1],'r-', Accuracy[:,0], Accuracy[:,2],'b-', Accuracy[:,0], Accuracy[:,3],'g-')
            plt.axhline(y=0.9,xmin=0,xmax=Accuracy[-1,0],color="r")
            plt.axhline(y=0.91,xmin=0,xmax=Accuracy[-1,0],color="g")
            plt.ylim(0, 1)
            plt.grid(visible=True)
            plt.legend(['iou','lp', 'iou&lp'])
            plt.savefig(os.path.join(self.OUTPUT,self.TIMESTAMP+'_p.jpg'))

            with open(os.path.join(self.OUTPUT, self.TIMESTAMP + "-_repr.csv"),'a') as f:
                f.write(str(self.STEP) + ',' + str(self.count_iou_lp/self.sample_num) + ',' + str(self.count_iou_lp/self.pred_num))
                f.write('\n')
            with open(os.path.join(self.OUTPUT, self.TIMESTAMP + "-_repr.csv"),'r',encoding='utf8') as f:
                txts = f.readlines()
                txts = [i.rstrip("\n") for i in txts]
                Accuracy = np.array([i.split(',') for i in txts]).astype(np.float32)
            plt.figure('repr')
            plt.plot(Accuracy[:,0], Accuracy[:,1],'r-', Accuracy[:,0], Accuracy[:,2],'b-')
            plt.axhline(y=0.8,xmin=0,xmax=Accuracy[-1,0],color="r")
            plt.axhline(y=0.9,xmin=0,xmax=Accuracy[-1,0],color="g")
            plt.ylim(0, 1)
            plt.grid(visible=True)
            plt.legend(['recall','precision'])
            plt.savefig(os.path.join(self.OUTPUT,self.TIMESTAMP+'_repr.jpg'))

            self.iou_list = self.iou_list.to('cpu').numpy()
            plt.figure('IoU')
            plt.hist(self.iou_list,50)
            plt.savefig(os.path.join(self.OUTPUT,self.TIMESTAMP+'_iou.jpg'))
            plt.clf()
    
    def iou(self, box, other_boxe):
        box_area = (box[2]-box[0])*(box[3]-box[1])
        other_boxes_area = (other_boxe[2]-other_boxe[0]) * (other_boxe[3]-other_boxe[1])
        x1 = torch.max(box[0],other_boxe[0])
        y1 = torch.max(box[1],other_boxe[1])
        x2 = torch.min(box[2],other_boxe[2])
        y2 = torch.min(box[3],other_boxe[3])
        Min = torch.zeros(1, device=box.device)
        w,h = torch.max(Min,x2-x1),torch.max(Min,y2-y1)
        overlap_area = w*h
        iou = overlap_area / (box_area+other_boxes_area-overlap_area)
        return iou


if __name__ == '__main__':

   
    CCPD_DIR = "/nas/origin/CCPD2018"
    SPLIT = 'data/split_2018'
    CRPD_DIR = "/nas/origin/CRPD"

    parameter = {
        'WORKSPACE':"./",
        'EPOCH' : 100,
        'LR' : 0.01,
        'BATCH_SIZE' : 8,
        'TRAIN_PRING_NUM': 10,
        'VAL_NUM': 2,
        'CHECKPOINT_NUM': -1,
        'CHECKPOINT': None,
        'LR_STEP_SIZE' : 5,
        'LR_STEP_GAMMA' : 0.5,
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 0,
        'BETAS':(0.9, 0.999),
    }

    train_CCPD_all = []
    with open(os.path.join(SPLIT,'train.txt'),'r',encoding='UTF-8') as f:
        txt = f.readlines()
        train_CCPD_all += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]
    train_CCPD_random = []
    train_CCPD_select = []
    for n in train_CCPD_all:
        img_label = n.split('/')[-1].rsplit('.', 1)[0].split('-')
        license_plate = img_label[4]
        license_plate = [int(i) for i in license_plate.split('_')]
        if license_plate[0]!=0:
            train_CCPD_select.append(n)
        else:
            train_CCPD_random.append(n)

    train_CCPD_green = []
    with open(os.path.join(SPLIT,'green_train.txt'),'r',encoding='UTF-8') as f:
        txt = f.readlines()
        train_CCPD_green += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]

    train_CRPD_normal = []
    train_CRPD_select = []
    train_CRPD_lack = []
    train_CRPD_two = []
    for i in os.listdir(CRPD_DIR+'/CRPD_single/train/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_single/train/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()[0].rstrip("\n")
                img_label = txt.split(' ')
            license_plate = img_label[-1]
            if len(license_plate)<7:
                raise Exception('len(license_plate)<7')
            pl = [CHARACTER.index(j) for j in license_plate]
            type = int(img_label[-2])
        except:
            continue
        if type==2:
            train_CRPD_two.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
        else:
            if any(x in list(license_plate) for x in provinces1):
                train_CRPD_lack.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
            elif any(x in list(license_plate) for x in provinces2):
                train_CRPD_select.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
            else:
                train_CRPD_normal.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
    
    for i in os.listdir(CRPD_DIR+'/CRPD_single/val/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_single/val/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()[0].rstrip("\n")
                img_label = txt.split(' ')
            license_plate = img_label[-1]
            if len(license_plate)<7:
                raise Exception('len(license_plate)<7')
            pl = [CHARACTER.index(j) for j in license_plate]
            type = int(img_label[-2])
        except:
            continue
        if type==2:
            train_CRPD_two.append(os.path.join(CRPD_DIR+'/CRPD_single/val/images', i))
        else:
            if any(x in list(license_plate) for x in provinces1):
                train_CRPD_lack.append(os.path.join(CRPD_DIR+'/CRPD_single/val/images', i))
            elif any(x in list(license_plate) for x in provinces2):
                train_CRPD_select.append(os.path.join(CRPD_DIR+'/CRPD_single/val/images', i))
            else:
                train_CRPD_normal.append(os.path.join(CRPD_DIR+'/CRPD_single/val/images', i))

    train_CRPD_multi = []
    for i in os.listdir(CRPD_DIR+'/CRPD_double/train/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_double/train/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()
                if len(txt)==0:
                    continue
                for j in txt:
                    img_label = j.rstrip("\n")
                    img_label = img_label.split(' ')
                    license_plate = img_label[-1]
                    if len(license_plate)<7:
                        raise Exception('len(license_plate)<7')
                    pl = [CHARACTER.index(j) for j in license_plate]
        except:
            continue
        if any(x in list(license_plate) for x in provinces1):
            for _ in range(20):
                train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_double/train/images', i))
        train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_double/train/images', i))
    for i in os.listdir(CRPD_DIR+'/CRPD_multi/train/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_multi/train/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()
                if len(txt)==0:
                    continue
                for j in txt:
                    img_label = j.rstrip("\n")
                    img_label = img_label.split(' ')
                    license_plate = img_label[-1]
                    if len(license_plate)<7:
                        raise Exception('len(license_plate)<7')
                    pl = [CHARACTER.index(j) for j in license_plate]
        except:
            continue
        if any(x in list(license_plate) for x in provinces1):
            for _ in range(20):
                train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_multi/train/images', i))
        train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_multi/train/images', i))

    for i in os.listdir(CRPD_DIR+'/CRPD_double/val/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_double/val/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()
                if len(txt)==0:
                    continue
                for j in txt:
                    img_label = j.rstrip("\n")
                    img_label = img_label.split(' ')
                    license_plate = img_label[-1]
                    if len(license_plate)<7:
                        raise Exception('len(license_plate)<7')
                    pl = [CHARACTER.index(j) for j in license_plate]
        except:
            continue
        if any(x in list(license_plate) for x in provinces1):
            for _ in range(20):
                train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_double/val/images', i))
        train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_double/val/images', i))
    for i in os.listdir(CRPD_DIR+'/CRPD_multi/val/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_multi/val/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()
                if len(txt)==0:
                    continue
                for j in txt:
                    img_label = j.rstrip("\n")
                    img_label = img_label.split(' ')
                    license_plate = img_label[-1]
                    if len(license_plate)<7:
                        raise Exception('len(license_plate)<7')
                    pl = [CHARACTER.index(j) for j in license_plate]
        except:
            continue
        if any(x in list(license_plate) for x in provinces1):
            for _ in range(20):
                train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_multi/val/images', i))
        train_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_multi/val/images', i))


    val_CCPD_select = []
    # with open(os.path.join(SPLIT,'val.txt'),'r') as f:
    #     txt = random.sample(f.readlines(), 10)
    #     val_CCPD_select += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]
    # for i in ['ccpd_challenge.txt', 'ccpd_db.txt', 'ccpd_fn.txt', 'ccpd_rotate.txt', 'ccpd_tilt.txt', 'ccpd_weather.txt']: #'ccpd_blur.txt', 
    #     with open(os.path.join(SPLIT,i),'r') as f:
    #         txt = random.sample(f.readlines(), 10)
    #         val_CCPD_select += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]

    val_CCPD_green = []
    # with open(os.path.join(SPLIT,'green_test.txt'),'r',encoding='UTF-8') as f:
    #     txt = random.sample(f.readlines(), 10)
    #     val_CCPD_green += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]
        
    val_CRPD_normal = []
    val_CRPD_select = []
    val_CRPD_lack = []
    val_CRPD_two = []
    for i in os.listdir(CRPD_DIR+'/CRPD_single/test/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_single/test/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()[0].rstrip("\n")
                img_label = txt.split(' ')
            license_plate = img_label[-1]
            pl = [CHARACTER.index(j) for j in license_plate]
            type = int(img_label[-2])
        except:
            continue
        if type==2:
            val_CRPD_two.append(os.path.join(CRPD_DIR+'/CRPD_single/test/images', i))
        else:
            if any(x in list(license_plate) for x in provinces1):
                val_CRPD_lack.append(os.path.join(CRPD_DIR+'/CRPD_single/test/images', i))
            elif any(x in list(license_plate) for x in provinces2):
                val_CRPD_select.append(os.path.join(CRPD_DIR+'/CRPD_single/test/images', i))
            else:
                val_CRPD_normal.append(os.path.join(CRPD_DIR+'/CRPD_single/test/images', i))
    val_CRPD_multi = []
    for i in os.listdir(CRPD_DIR+'/CRPD_double/test/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_double/test/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()
                if len(txt)==0:
                    continue
                for j in txt:
                    img_label = j.rstrip("\n")
                    img_label = img_label.split(' ')
                    license_plate = img_label[-1]
                    pl = [CHARACTER.index(j) for j in license_plate]
        except:
            continue
        val_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_double/test/images', i))
    for i in os.listdir(CRPD_DIR+'/CRPD_multi/test/images'):
        try:
            img_label = os.path.join(CRPD_DIR+'/CRPD_multi/test/images', i).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            with open(img_label, 'r', encoding='UTF-8') as f:
                txt = f.readlines()
                if len(txt)==0:
                    continue
                for j in txt:
                    img_label = j.rstrip("\n")
                    img_label = img_label.split(' ')
                    license_plate = img_label[-1]
                    pl = [CHARACTER.index(j) for j in license_plate]
        except:
            continue
        val_CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_multi/test/images', i))
    
    imgSize = (1024, 1024)
    train_dataset = MyDataset(train_CCPD_random, train_CCPD_select*2, train_CCPD_green*2, 
                              train_CRPD_normal, train_CRPD_select, train_CRPD_lack*100, 
                              train_CRPD_two*30, train_CRPD_multi*5, imgSize, CCPD_random_num=30000, 
                              replace_rate=[0.7, 1, 0.7, 0.6, 0.7, 0.9, 0.1, 0.7], replace=True, transform=True)
    val_dataset = MyDataset([], val_CCPD_select, val_CCPD_green, val_CRPD_normal, val_CRPD_select, val_CRPD_lack, val_CRPD_two,val_CRPD_multi, imgSize)

    mask_x = (np.array([[i for i in range(train_dataset.grid_det)]]*train_dataset.grid_det) + 0.5)*imgSize[0]/train_dataset.grid_det
    mask_y = (np.array([[i]*train_dataset.grid_det for i in range(train_dataset.grid_det)]) + 0.5)*imgSize[1]/train_dataset.grid_det
    mask = torch.from_numpy(np.stack([mask_x, mask_y], axis=2))
    e = Experiment(**parameter)
    e.set_seed()
    e.print_parameter()
    e.load_dataLoader(train_dataset, val_dataset, num_workers=10)
    e.load_model()
    e.train()