import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import random
from PIL import Image

import utils
from utils import resize, warp, replace, blur, add_random_erase, add_random_gauss_brightness, iou, crop, clean, provinces1, provinces2

CHARACTER = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏",
             "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", 
             "琼", "渝", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
             "新", "A",  "B",  "C",  "D",  "E",  "F", "G",  "H",  "J", 
             "K",  "L",  "M",  "N",  "P",  "Q",  "R", "S",  "T",  "U", 
             "V",  "W",  "X",  "Y",  "Z",  "0",  "1", "2",  "3",  "4", 
             "5",  "6",  "7",  "8",  "9",  "港", "澳", "使", "领", "学", 
             "警", "挂", ""]

PROVINCES = ["皖", "沪", "津", "渝", "冀", 
             "晋", "蒙", "辽", "吉", "黑", 
             "苏", "浙", "京", "闽", "赣", 
             "鲁", "豫", "鄂", "湘", "粤", 
             "桂", "琼", "川", "贵", "云", 
             "藏", "陕", "甘", "青", "宁", 
             "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 
             'F', 'G', 'H', 'J', 'K', 
             'L', 'M', 'N', 'P', 'Q', 
             'R', 'S', 'T', 'U', 'V', 
             'W', 'X', 'Y', 'Z', 'O']
ADS =  ['A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'J', 'K', 
        'L', 'M', 'N', 'P', 'Q', 
        'R', 'S', 'T', 'U', 'V', 
        'W', 'X', 'Y', 'Z', '0', 
        '1', '2', '3', '4', '5', 
        '6', '7', '8', '9', 'O']

'''
0 for blue plates
1 for yellow and single-line plates
2 for yellow and double-lines plates
3 for white plates

blue(yellow) / black / white_army
black_shi 
black_ling
green_car / green_truck
white
double
'''

class MyDataset(Dataset):
    def __init__(self, CCPD_random:list, CCPD_select:list, CCPD_green:list, 
                 CRPD_normal:list, CRPD_select:list, CRPD_lack:list, CRPD_two:list, CRPD_multi:list, imgSize:tuple, CCPD_random_num = 5000,
                 replace_rate=[0.5, 1, 0.5, 0.5, 1, 0.1, 0.1, 0.5], replace=False, transform=False):
        '''
        CCPD_random 皖
        CCPD_select 其它
        CCPD_green  绿牌
        CRPD_normal 川
        CRPD_select 京冀吉晋津浙湘琼甘皖粤藏豫贵闽陕青鲁辽  新蒙黑
        CRPD_lack   云宁桂沪渝苏赣鄂
        CRPD_two    双层
        CRPD_multi  多牌
        '''

        self.CCPD_random = CCPD_random
        self.CCPD_select = CCPD_select
        self.CCPD_green = CCPD_green
        self.CRPD_normal = CRPD_normal
        self.CRPD_select = CRPD_select
        self.CRPD_lack = CRPD_lack
        self.CRPD_two = CRPD_two
        self.CRPD_multi = CRPD_multi
        self.image_list = CCPD_select + CCPD_green + CRPD_normal + CRPD_select + CRPD_lack + CRPD_two + CRPD_multi
        self.img_size = imgSize  #w,h
        self.grid_det = 64
        self.grid_rec = 16
        self.replace_rate = replace_rate
        self.lp_at_size = (imgSize[0]//4, imgSize[1]//4)
        self.ch_at_size = self.lp_at_size
        self.CCPD_random_num = CCPD_random_num if len(CCPD_random)>CCPD_random_num else len(CCPD_random)
        self.replace = replace
        self.transform = transform
        self.tran_pil = transforms.Compose([
            transforms.ToPILImage(),
            ])
        self.tran_color = transforms.RandomApply([
            transforms.ColorJitter(0.7, 0.4, 0.4)
            ], p=0.5)
        self.tran_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.list_num = [self.CCPD_random_num, len(self.CCPD_select), len(self.CCPD_green), len(self.CRPD_normal), len(self.CRPD_select), len(self.CRPD_lack), len(self.CRPD_two), len(self.CRPD_multi)]
        
        print(f' \
                CCPD_random:{self.list_num[0]}, \n \
                CCPD_select:{self.list_num[1]}, \n \
                CCPD_green: {self.list_num[2]}, \n \
                CRPD_normal:{self.list_num[3]}, \n \
                CRPD_select:{self.list_num[4]}, \n \
                CRPD_lack:  {self.list_num[5]}, \n \
                CRPD_two:   {self.list_num[6]}, \n \
                CRPD_multi: {self.list_num[7]}')
        if replace:
            print('remain')
            print(f' \
                CCPD_random:{self.list_num[0]*self.replace_rate[0]}, \n \
                CCPD_select:{self.list_num[1]*self.replace_rate[1]}, \n \
                CCPD_green: {self.list_num[2]*self.replace_rate[2]}, \n \
                CRPD_normal:{self.list_num[3]*self.replace_rate[3]}, \n \
                CRPD_select:{self.list_num[4]*self.replace_rate[4]}, \n \
                CRPD_lack:  {self.list_num[5]*self.replace_rate[5]}, \n \
                CRPD_two:   {self.list_num[6]*self.replace_rate[6]}, \n \
                CRPD_multi: {self.list_num[7]*self.replace_rate[7]}')

    def __len__(self):
        return self.CCPD_random_num + len(self.CCPD_select) + len(self.CCPD_green) + len(self.CRPD_normal) + len(self.CRPD_select) + len(self.CRPD_lack) + len(self.CRPD_two) + len(self.CRPD_multi)

    def __getitem__(self, index):
        #TODO get the original lable
        if index<self.CCPD_random_num: #TODO from CCPD_random
            img_name = random.sample(self.CCPD_random, 1)[0]
            img_label = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')
            [leftUp, rightDown] = [[int(j) for j in i.split('&')] for i in img_label[2].split('_')]
            vertices_label = [[int(j) for j in i.split('&')] for i in img_label[3].split('_')]
            license_plate = [int(i) for i in img_label[4].split('_')]
            lp = PROVINCES[license_plate[0]]
            lp += ALPHABETS[license_plate[1]]
            for j in range(2,len(license_plate)):
                lp += ADS[license_plate[j]]
            bg_color = 'blue'

            lurd_list = [[leftUp, rightDown]]
            vertices_label_list = [vertices_label]
            lp_list = [lp]
            bg_color_list = [bg_color]

        else:
            img_name = self.image_list[index-self.CCPD_random_num]
            img_label = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')
            if len(img_label)>1: #TODO from CCPD_select & CCPD_green
                [leftUp, rightDown] = [[int(j) for j in i.split('&')] for i in img_label[2].split('_')]
                vertices_label = [[int(j) for j in i.split('&')] for i in img_label[3].split('_')]
                license_plate = [int(i) for i in img_label[4].split('_')]
                lp = PROVINCES[license_plate[0]]
                lp += ALPHABETS[license_plate[1]]
                for j in range(2,len(license_plate)):
                    lp += ADS[license_plate[j]]
                if len(license_plate)==8:
                    bg_color = 'green_car'
                else:
                    bg_color = 'blue'

                lurd_list = [[leftUp, rightDown]]
                vertices_label_list = [vertices_label]
                lp_list = [lp]
                bg_color_list = [bg_color]
                
            else:   #TODO from CRPD_normal & CRPD_select  & CRPD_lack & CRPD_two & CRPD_multi
                lurd_list = []
                vertices_label_list = []
                lp_list = []
                bg_color_list = []
                
                img_label = img_name.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
                with open(img_label, 'r', encoding='UTF-8') as f:
                    txt = f.readlines()
                    for i in txt:
                        img_label = i.rstrip("\n")
                        img_label = img_label.split(' ')
                        vertices_label = [int(j) for j in img_label[:8]]
                        if vertices_label[3] < vertices_label[7]:
                            vertices_label = [vertices_label[4:6], vertices_label[6:8], vertices_label[0:2], vertices_label[2:4]]
                        else:
                            vertices_label = [vertices_label[4:6], vertices_label[2:4], vertices_label[0:2], vertices_label[6:8]]
                        tmp = np.array(vertices_label)
                        minx = np.min(tmp[:,0])
                        maxx = np.max(tmp[:,0])
                        miny = np.min(tmp[:,1])
                        maxy = np.max(tmp[:,1])
                        leftUp, rightDown = [minx, miny], [maxx, maxy]
                        lp = img_label[-1]
                        try:
                            if lp[0]==lp[-1]:
                                lp = lp[:-1]
                        except:
                            print(i)
                            print(img_name)
                        if int(img_label[-2])==2:
                            bg_color = 'double'
                        else:
                            bg_color = 'blue'

                        lurd_list.append([leftUp, rightDown])
                        vertices_label_list.append(vertices_label)
                        lp_list.append(lp)
                        bg_color_list.append(bg_color)

        img = cv2.imread(img_name)

        #TODO replace the lp
        if self.transform and self.replace:
            for i in range(len(lp_list)):
                flag = False
                if index<self.list_num[0] and random.random()>self.replace_rate[0]: #from CCPD_random
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'norm')
                    flag = True
                elif self.list_num[0]<=index<sum(self.list_num[:2]) and random.random()>self.replace_rate[1]: #from CCPD_select
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'norm')
                    flag = True
                elif sum(self.list_num[:2])<=index<sum(self.list_num[:3]) and random.random()>self.replace_rate[2]: #from CCPD_green
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'green')
                    flag = True
                elif sum(self.list_num[:3])<=index<sum(self.list_num[:4]) and random.random()>self.replace_rate[3]: #from CRPD_normal
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'multi', 0.5)
                    flag = True
                elif sum(self.list_num[:4])<=index<sum(self.list_num[:5]) and random.random()>self.replace_rate[4]: #from CRPD_select
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'multi', 0.5)
                    flag = True
                elif sum(self.list_num[:5])<=index<sum(self.list_num[:6]) and random.random()>self.replace_rate[5]: #from CRPD_lack
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'lack', 0.5)
                    flag = True
                elif sum(self.list_num[:6])<=index<sum(self.list_num[:7]) and random.random()>self.replace_rate[6]: #from CRPD_two
                    img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], lurd_list[i][0], lurd_list[i][1], 'double', 0.5)
                    flag = True
                elif sum(self.list_num[:7])<=index<sum(self.list_num): #from CRPD_mutil
                    if bg_color_list[i]!='double':
                        if any(x in list(lp_list[i]) for x in provinces1):
                            # from provinces1
                            if random.random()>self.replace_rate[5]:
                                img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], leftUp, rightDown, 'lack', 0.5)
                                flag = True
                        elif any(x in list(lp_list[i]) for x in provinces2):
                            # from provinces2
                            if random.random()>self.replace_rate[4]:
                                img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], leftUp, rightDown, 'multi', 0.5)
                                flag = True
                        elif random.random()>self.replace_rate[7]:
                            # from chuan
                            img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], leftUp, rightDown, 'multi', 0.5)
                            flag = True
                    else:
                        if random.random()>self.replace_rate[6]:
                            img, vertices_label, leftUp, rightDown, lp, bg_color = replace(img, vertices_label_list[i], leftUp, rightDown, 'double', 0.5)
                            flag = True
                if flag:   
                    lurd_list[i] = [leftUp, rightDown]
                    vertices_label_list[i] = vertices_label
                    lp_list[i] = lp
                    bg_color_list[i] = bg_color
            
        # if sum(self.list_num[:3])<=index: #from CRPD
        #     img, lurd_list, vertices_label_list = crop(img, lurd_list, vertices_label_list)
                    
        # if self.transform and sum(self.list_num[:3])<=index<sum(self.list_num[:5]) and random.random()<0.5:
        #     img = clean(img, lurd_list)

        if self.transform:
            if index<sum(self.list_num[:3]): #from CCPD
                if random.random()<0.5:
                    img, lurd_list, vertices_label_list = resize(img, lurd_list, vertices_label_list)
            elif sum(self.list_num[:6])<=index<sum(self.list_num[:7]):
                if random.random()<0.8:
                    img, lurd_list, vertices_label_list = resize(img, lurd_list, vertices_label_list)
            else:
                if random.random()<0.2:
                    img, lurd_list, vertices_label_list = resize(img, lurd_list, vertices_label_list)

        if self.transform:
            if index<sum(self.list_num[:3]): #from CCPD
                if random.random()<0.7:
                    img, lurd_list, vertices_label_list = warp(img, lurd_list, vertices_label_list)
            else:
                if random.random()<0.3:
                    img, lurd_list, vertices_label_list = warp(img, lurd_list, vertices_label_list)

        height, width, _ = img.shape
        # origin_lurd = np.array(lurd_list)

        size =  height if height>width else width
        img2 = np.zeros((size, size, 3)).astype("uint8")
        if self.transform:
            if height == size:
                x, y = random.randint(0, size-width), 0
            else:
                x, y = 0, random.randint(0, size-height)
        else:
            if height == size:
                x, y = (size-width)//2, 0
            else:
                x, y = 0, (size-height)//2
        img2[y:y+height,x:x+width,:] = img
        for vertices_label in vertices_label_list:
            for i in vertices_label:
                i[0] = i[0] + x
                i[1] = i[1] + y
        for lurd in lurd_list:
            for i in lurd:
                i[0] = i[0] + x
                i[1] = i[1] + y

        # todo create attention map of lp
        lp_at = np.zeros((self.lp_at_size[0],self.lp_at_size[1])).astype("uint8")
        lp_at_det = []
        lp_at_rec = []
        lp_at_rec_facal = np.zeros((self.grid_rec, self.grid_rec))
        new_vertices_list = []
        for vertices_label in vertices_label_list:
            new_vertices = []
            for i in vertices_label:
                new_vertices.append([round(i[0] * (self.lp_at_size[0]/size)), round(i[1] * (self.lp_at_size[1]/size))])
            new_vertices_list.append(new_vertices)

            pts = np.array(new_vertices, np.int32)
            pts = pts.reshape((-1,1,2))  
            cv2.fillPoly(lp_at,[pts],255,lineType=cv2.LINE_AA)
            lp_at_det.append(cv2.fillPoly(np.zeros((self.lp_at_size[0],self.lp_at_size[1])).astype("uint8"),[pts],255,lineType=cv2.LINE_AA))
            lp_at_rec.append(cv2.fillPoly(np.zeros((self.lp_at_size[0],self.lp_at_size[1])).astype("uint8"),[pts],255,lineType=cv2.LINE_AA))

        lp_at = lp_at.astype('float32')
        lp_at /= 255.0
        lp_at = lp_at>0.5
        lp_at = lp_at.astype('float32')

        for i in range(len(lp_at_det)):
            lp_at_det[i] = F.adaptive_avg_pool2d(torch.from_numpy(lp_at_det[i]).unsqueeze_(dim=0).unsqueeze_(dim=0).float(), (self.grid_det, self.grid_det)).squeeze_(dim=0).squeeze_(dim=0).numpy()
            lp_at_rec[i] = F.adaptive_avg_pool2d(torch.from_numpy(lp_at_rec[i]).unsqueeze_(dim=0).unsqueeze_(dim=0).float(), (self.grid_rec, self.grid_rec)).squeeze_(dim=0).squeeze_(dim=0).numpy()
            lp_at_rec_facal += (lp_at_rec[i]>0)*abs((1.001-0.001*np.sum((lp_at_rec[i]>0))))

        # todo create attention map of ch
        ch_at = self.create_character_attention(new_vertices_list, bg_color_list, self.lp_at_size)

        img = cv2.resize(img2, (self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            if random.random()<0.2:
                img = blur(img)
            if random.random()<0.2:
                img = add_random_erase(img)
            if random.random()<0.2:
                img = add_random_gauss_brightness(img)     
        # BGR -> RGB
        img = img[:, :,::-1]
        img = self.tran_pil(img)
        if self.transform:
            img = self.tran_color(img)
        img = self.tran_tensor(img)

        # bboxs [4,4,4] r,t,l,b
        bboxs = np.zeros((self.grid_det, self.grid_det, 4))
        lp = lp = np.zeros((self.grid_rec, self.grid_rec, 8))
        lurds = np.zeros((self.grid_det, self.grid_det, 4)).astype('int32')

        # resize coordinate to img_size
        for lurd in lurd_list:
            for i in lurd:
                i[0] = i[0] * (self.img_size[0]/size)
                i[1] = i[1] * (self.img_size[1]/size)

        grid_width = np.ceil(self.img_size[0]/self.grid_det)
        grid_heigh = np.ceil(self.img_size[1]/self.grid_det)
        
        lp_lurd = []
        for i in range(len(lp_list)):

            leftUp, rightDown = lurd_list[i]
            license_plate = [CHARACTER.index(j) for j in lp_list[i]]
            while len(license_plate)<8:
                license_plate.append(len(CHARACTER)-1)
            license_plate = np.array(license_plate)
            lp_lurd.append(','.join(np.round(np.array(leftUp+rightDown)).astype('int32').astype('str'))+'-'+','.join(license_plate.astype('str')))

            index_row, index_col = lp_at_det[i].nonzero()
            for j in range(len(index_row)):
                x = (index_col[j]+0.5)*grid_width 
                y = (index_row[j]+0.5)*grid_heigh 

                target = np.array([(x-leftUp[0])/self.img_size[0],(y-leftUp[1])/self.img_size[1],(rightDown[0]-x)/self.img_size[0],(rightDown[1]-y)/self.img_size[1]])
                if (target>0).all():
                    bboxs[index_row[j], index_col[j], :] = target
                    lurds[index_row[j], index_col[j], :] = np.round(np.array(leftUp+rightDown)).astype('int32')

            index_row, index_col = lp_at_rec[i].nonzero()
            for j in range(len(index_row)):
                lp[index_row[j], index_col[j], :] = license_plate
        
        lp = lp.astype('int64')
        lp_lurd = ';'.join(lp_lurd)
        return img, lp_at, ch_at, bboxs, lp, lurds, lp_at_rec_facal, lp_lurd

    def create_character_attention(self, new_vertices_list, bg_color_list, im_size):

        character_attention_map = []
        for _ in range(8):
            character_attention_map.append(np.zeros(im_size).astype("uint8"))
            
        for img_index in range(len(bg_color_list)):
            new_vertices_label = new_vertices_list[img_index]
            bg_color = bg_color_list[img_index]
        
            if bg_color=='double':
                at_index = 6
                v4_x = np.linspace(new_vertices_label[0][0],new_vertices_label[3][0],23)[12]
                v4_y = np.linspace(new_vertices_label[0][1],new_vertices_label[3][1],23)[12]
                v5_x = np.linspace(new_vertices_label[1][0],new_vertices_label[2][0],23)[12]
                v5_y = np.linspace(new_vertices_label[1][1],new_vertices_label[2][1],23)[12]
                xd = np.linspace(new_vertices_label[0][0],new_vertices_label[1][0],440)
                yd = np.linspace(new_vertices_label[0][1],new_vertices_label[1][1],440)
                xu = np.linspace(v4_x,v5_x,440)
                yu = np.linspace(v4_y,v5_y,440)
                index = [20,100,180,260,340,420]
                for i in range(len(index)-1):
                    pts = np.array([[xd[index[i]],yd[index[i]]],[xd[index[i+1]],yd[index[i+1]]],[xu[index[i+1]],yu[index[i+1]]],[xu[index[i]],yu[index[i]]]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    pts = np.round(pts).astype('int32')
                    cv2.fillPoly(character_attention_map[at_index],[pts],255,lineType=cv2.LINE_AA)
                    at_index -=1
                    
                xd = np.linspace(v4_x,v5_x,440)
                yd = np.linspace(v4_y,v5_y,440)
                xu = np.linspace(new_vertices_label[3][0],new_vertices_label[2][0],440)
                yu = np.linspace(new_vertices_label[3][1],new_vertices_label[2][1],440)
                index = [105,195,245,335]
                jump_index = 1
                for i in range(len(index)-1):
                    if i ==jump_index:
                        continue
                    pts = np.array([[xd[index[i]],yd[index[i]]],[xd[index[i+1]],yd[index[i+1]]],[xu[index[i+1]],yu[index[i+1]]],[xu[index[i]],yu[index[i]]]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    pts = np.round(pts).astype('int32')
                    cv2.fillPoly(character_attention_map[at_index],[pts],255,lineType=cv2.LINE_AA)
                    at_index -=1
            else:
                if bg_color in ['green_car', 'green_truck']:
                    at_index = 7
                    xd = np.linspace(new_vertices_label[0][0],new_vertices_label[1][0],480)
                    yd = np.linspace(new_vertices_label[0][1],new_vertices_label[1][1],480)
                    xu = np.linspace(new_vertices_label[3][0],new_vertices_label[2][0],480)
                    yu = np.linspace(new_vertices_label[3][1],new_vertices_label[2][1],480)
                    index = [11,63,115,167,219,271,323,363,415,469]
                    jump_index = 6
                else:
                    at_index = 6
                    xd = np.linspace(new_vertices_label[0][0],new_vertices_label[1][0],440)
                    yd = np.linspace(new_vertices_label[0][1],new_vertices_label[1][1],440)
                    xu = np.linspace(new_vertices_label[3][0],new_vertices_label[2][0],440)
                    yu = np.linspace(new_vertices_label[3][1],new_vertices_label[2][1],440)
                    if bg_color=='black_shi':
                        index = [10,67,124,181,238,260,317,374,431]
                        jump_index = 4
                    elif bg_color=='black_ling':
                        index = [10,67,124,181,203,260,317,374,431]
                        jump_index = 3
                    elif bg_color=='white':
                        index = [10,67,124,181,238,295,352,374,431]
                        jump_index = 6
                    else:
                        index = [10,67,124,181,238,295,317,374,431]
                        jump_index = 5
                for i in range(len(index)-1):
                    if i ==jump_index:
                        continue
                    pts = np.array([[xd[index[i]],yd[index[i]]],[xd[index[i+1]],yd[index[i+1]]],[xu[index[i+1]],yu[index[i+1]]],[xu[index[i]],yu[index[i]]]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    pts = np.round(pts).astype('int32')
                    at = np.zeros(im_size).astype("uint8")
                    cv2.fillPoly(character_attention_map[at_index],[pts],255,lineType=cv2.LINE_AA)
                    at_index -=1

        for i in range(8):
            character_attention_map[i] = cv2.resize(character_attention_map[i], self.ch_at_size).astype('float32')
            character_attention_map[i] /= 255.0
            character_attention_map[i] = character_attention_map[i]>0.5
            character_attention_map[i] = character_attention_map[i].astype('float32')
            
        return np.array(character_attention_map)

if __name__ == '__main__':

    CCPD_DIR = "E:/BaiduNetdiskDownload/CCPD2019"
    CRPD_DIR = "E:/BaiduNetdiskDownload/CRPD_all"
    SPLIT = 'E:/Files/workspace/data/ALPR/CCPD/split_2019'

    CCPD_all = []
    with open(os.path.join(SPLIT,'train.txt'),'r',encoding='UTF-8') as f:
        txt = f.readlines()
        CCPD_all += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]
    CCPD_random = []
    CCPD_select = []
    for n in CCPD_all:
        img_label = n.split('/')[-1].rsplit('.', 1)[0].split('-')
        license_plate = img_label[4]
        license_plate = [int(i) for i in license_plate.split('_')]
        if license_plate[0]!=0:
            CCPD_select.append(n)
        else:
            CCPD_random.append(n)

    CCPD_green = []
    with open(os.path.join(SPLIT,'green_train.txt'),'r',encoding='UTF-8') as f:
        txt = f.readlines()
        CCPD_green += [os.path.join(CCPD_DIR,i.rstrip("\n")) for i in txt]

    CRPD_normal = []
    CRPD_select = []
    CRPD_lack = []
    CRPD_two = []
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
            CRPD_two.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
        else:
            if any(x in list(license_plate) for x in provinces1):
                CRPD_lack.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
            elif any(x in list(license_plate) for x in provinces2):
                CRPD_select.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
            else:
                CRPD_normal.append(os.path.join(CRPD_DIR+'/CRPD_single/train/images', i))
        
    CRPD_multi = []
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
        CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_double/train/images', i))
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
        CRPD_multi.append(os.path.join(CRPD_DIR+'/CRPD_multi/train/images', i))
        
    
    imgSize = (1024, 1024) #w,h
    train_dataset = MyDataset(CCPD_random, (CCPD_select*2)[:1], (CCPD_green*2)[:1], 
                              CRPD_normal[:-10], CRPD_select[:-10], (CRPD_lack*10)[:-10], (CRPD_two*30)[:-10], (CRPD_multi*5)[:-10], imgSize, 
                              CCPD_random_num=1, replace_rate=[0.5, 1, 0.5, 0.5, 0.5, 0.9, 0.1, 0.3], replace=True, transform=True)
    train_dataloader = DataLoader(train_dataset, 1, shuffle=True)

    a = []

    for batch, data in enumerate(train_dataloader):    #img, lp_at, ch_at,bboxs, lp, lurds, sizexy, img_name
        if batch>2000:
            break
        if batch%100==0:
            print('batch', batch)
        
        # print('batch', batch)

        img, lp_at, ch_at, bboxs, lp, lurds, lp_at_rec_facal, lp_lurd= data

        # print(img.shape)
        # print(lp_at.shape)
        # print(ch_at.shape)
        # print(bboxs.shape)
        # print(lp.shape)
        # print(lurds.shape)
        # print(lp_at_rec_facal.shape)
        # print(sizexy.shape)

        # print(lp_at_rec_facal)
  
        # img = img.transpose(1, 3).transpose(1, 2)  # b*C*H*W --> b*H*W*C
        # img = np.array(img) * 255
        # img = np.ascontiguousarray(img.astype('uint8')[0])

        # lp_at = lp_at.unsqueeze(dim=1)
        # at = torch.cat([lp_at, ch_at],dim=1)
        # at = at[0]

        # for i in range(at.shape[0]):

        #     at_i = at[i]
        #     at_i = np.array(at_i) * 255
        #     at_i = np.stack([at_i, at_i, at_i], axis=-1).astype('uint8')
        #     cv2.imwrite('output/'+str(batch)+'-'+str(i)+'.jpg', at_i)
        #     at_i = cv2.resize(np.array(at[i]), imgSize)     
        #     im = img*np.expand_dims(at_i, axis=2)
        #     im = im.astype('uint8')
        #     im = Image.fromarray(im).convert('RGB')
        #     im.save('output/'+str(batch)+'_'+str(i)+'.jpg')

        lp_lurd_list = lp_lurd[0].split(';')
        lp_list = []
        lurd_list = []
        for i in lp_lurd_list:
            try:
                lurd, lp = i.split('-')
            except:
                print(lp_lurd_list)
                exit()
            lp_list.append(np.array(lp.split(',')).astype('int32'))
            lurd_list.append(np.array(lurd.split(',')).astype('int32'))
        # print(lp_list)
        # print(lurd_list)
        for i in lp_list:
            a.extend(list(i))
        # for i in range(len(lp_list)):
        #     cv2.rectangle(img, lurd_list[i][:2], lurd_list[i][2:],(255,0,0),2)
        # # # for i in np.linspace(0, img.shape[0], 65).astype('int32'):
        # #     cv2.line(img, (0, i), (img.shape[1], i), (0, 255, 0), 1)
        # # for i in np.linspace(0, img.shape[1], 65).astype('int32'):
        # #     cv2.line(img, (i, 0), (i, img.shape[0]), (0, 255, 0), 1)

        # for i in np.linspace(0, img.shape[0], 9).astype('int32'):
        #     cv2.line(img, (0, i), (img.shape[1], i), (0, 0, 255), 1)
        # for i in np.linspace(0, img.shape[1], 9).astype('int32'):
        #     cv2.line(img, (i, 0), (i, img.shape[0]), (0, 0, 255), 1)
        
        # img = Image.fromarray(img).convert('RGB')
        # img.save('output/'+str(batch)+'.jpg')

        # exit()
    
    n, indices = np.unique(a, return_counts=True)
    print(n)
    print(indices)
