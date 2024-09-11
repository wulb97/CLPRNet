from model import CLPRNet
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

CHARACTER = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏",
             "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", 
             "琼", "渝", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
             "新", "A",  "B",  "C",  "D",  "E",  "F", "G",  "H",  "J", 
             "K",  "L",  "M",  "N",  "P",  "Q",  "R", "S",  "T",  "U", 
             "V",  "W",  "X",  "Y",  "Z",  "0",  "1", "2",  "3",  "4", 
             "5",  "6",  "7",  "8",  "9",  "港", "澳", "使", "领", "学", 
             "警", "挂", ""]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (1024, 1024)

font_size = 30
font = ImageFont.truetype('resource/msyh.ttc',font_size,encoding='utf-8')

model = CLPRNet()   
model = model.to(DEVICE) 
model.load_state_dict(torch.load('resource/CLPRNet.pth', map_location=DEVICE))
model.eval()

if not os.path.exists('output'):
    os.makedirs('output')

tran = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

def NMS(boxes, C = 0.3):
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

def inference(src, image_list):
    
    grid = 64
    mask_x = (np.array([[i for i in range(grid)]]*grid) + 0.5)*img_size[0]/grid
    mask_y = (np.array([[i]*grid for i in range(grid)]) + 0.5)*img_size[1]/grid
    mask = torch.from_numpy(np.stack([mask_x, mask_y], axis=2))
    x_mask = mask[:,:,0].to(DEVICE).unsqueeze_(dim=2)
    y_mask = mask[:,:,1].to(DEVICE).unsqueeze_(dim=2)

    for img_name in image_list:
        print(img_name)
        org_img = cv2.imread(os.path.join(src, img_name))
        #! normalize img
        height, width, _ = org_img.shape
        size =  height if height>width else width
        img2 = np.zeros((size, size, 3)).astype("uint8")
        if height == size:
            y = 0
            x = (size-width)//2
        else:
            x = 0
            y = (size-height)//2
        img2[y:y+height,x:x+width,:] = org_img
        img = cv2.resize(img2, img_size)

        #! inference
        inputs = img[:, :,::-1]
        inputs = tran(inputs)
        inputs = inputs.unsqueeze(dim=0)
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            y_detection, y_recognition,_,_ = model(inputs)
        y_recognition = y_recognition.repeat_interleave(int(y_detection.shape[1]/y_recognition.shape[1]), dim=1)
        y_recognition = y_recognition.repeat_interleave(int(y_detection.shape[2]/y_recognition.shape[2]), dim=2)

        #! save result
        for index in range(y_detection.shape[0]):
           
            l, t, r, b, c = torch.split(y_detection[index,:,:,:5], 1, dim=-1)
            l = x_mask - l*inputs.shape[3]  
            t = y_mask - t*inputs.shape[2]
            r = x_mask + r*inputs.shape[3]
            b = y_mask + b*inputs.shape[2]
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
            # ch = torch.mean(torch.stack([torch.max(ch1, dim=2)[0], torch.max(ch2, dim=2)[0], torch.max(ch3, dim=2)[0], torch.max(ch4, dim=2)[0], torch.max(ch5, dim=2)[0], torch.max(ch6, dim=2)[0], torch.max(ch7, dim=2)[0], torch.max(ch8, dim=2)[0]],dim=2), dim=2)
            c = c.squeeze_(dim=2) * ch

            out = torch.flatten(torch.concat([c.unsqueeze_(dim=2), l, t, r, b, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8], dim=2), start_dim=0, end_dim=1)
            out = out[torch.where(out[:,0]>0.3)]
            out = NMS(out, 0.3)

            preb_lurd_list = []
            preb_pl_list = []
            preb_c = []
            for i in out:
                lurd = torch.tensor([i[1], i[2], i[3], i[4]]).cpu().numpy()  
                lurd[0] = lurd[0]*size/img_size[0] - x
                lurd[1] = lurd[1]*size/img_size[1] - y
                lurd[2] = lurd[2]*size/img_size[0] - x
                lurd[3] = lurd[3]*size/img_size[1] - y

                lp_pred = []
                for j in range(8):
                    lp_pred.append(torch.argmax(i[5+j*73:5+(j+1)*73]))
                lp_pred = torch.stack(lp_pred).cpu().numpy()
                lp=''
                for j in range(8):
                    lp += CHARACTER[lp_pred[j]]
                preb_lurd_list.append(lurd.astype('int32'))
                preb_pl_list.append(lp)
                preb_c.append(round(float(i[0].cpu().numpy()),3))
                
            for i in preb_lurd_list:
                cv2.rectangle(org_img,i[:2],i[2:],(0,0,255),2)
                
            org_img = org_img[:, :,::-1]
            org_img = Image.fromarray(org_img.astype('uint8')).convert('RGB')
            draw = ImageDraw.Draw(org_img)

            for i in range(len(preb_pl_list)):
                label_size = int(draw.textlength(preb_pl_list[i]+'_'+str(preb_c[i]), font))
                draw.rectangle([(preb_lurd_list[i][0],preb_lurd_list[i][1]-font_size), (preb_lurd_list[i][0]+label_size,preb_lurd_list[i][1])], fill='red')
                draw.text(xy=(preb_lurd_list[i][0],preb_lurd_list[i][1]-int(font_size*1.25)),text=preb_pl_list[i]+'_'+str(preb_c[i]),fill=(255,255,255),font=font, embedded_color=True)

            org_img.save('output/'+img_name)

if __name__ == '__main__':

    src = 'image'
    inference(src, os.listdir(src))