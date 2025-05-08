import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, lr_scheduler
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random


class BaseExperiment():
    '''
    must overload:
    load_model, load_optimizer, load_scheduler  -> set model, optimizer, scheduler
    forward, loss -> set forward & loss function
    before_val, evaluate, after_val -> set evaluate function (before val -> evaluate -> after val)
    
    Step: 
    1. set parameter
    2. load dataLoader 
    3. load model
    4. train / val /test
    '''

    def __init__(self, **parameter) -> None:
        self.mode = ''
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TIMESTAMP = time.strftime("%Y-%m-%d %H%M%S",time.localtime())
        self.WORKSPACE = parameter['WORKSPACE']
        self.OUTPUT = os.path.join(self.WORKSPACE, self.TIMESTAMP)
        self.START_EPOCH = 1
        self.EPOCH:int = parameter['EPOCH'] if 'EPOCH' in parameter else 1
        self.BATCH_SIZE:int = parameter['BATCH_SIZE'] if 'BATCH_SIZE' in parameter else 1
        self.LR:float = parameter['LR'] if 'LR' in parameter else 0.1
        self.TRAIN_PRING_NUM:int = parameter['TRAIN_PRING_NUM'] if 'TRAIN_PRING_NUM' in parameter else 10
        self.VAL_NUM:int = parameter['VAL_NUM'] if 'VAL_NUM' in parameter else 1
        self.CHECKPOINT_NUM:int = parameter['CHECKPOINT_NUM'] if 'CHECKPOINT_NUM' in parameter else 0
        self.CHECKPOINT =  parameter['CHECKPOINT'] if 'CHECKPOINT' in parameter else None
        self.STEP = 0
    
    def print(self, x):
        print(x)

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def print_parameter(self):
        self.print(f"{'*'*10} parameter {'*'*10}")
        for i in self.__dict__:
            if i.isupper():
                self.print(f"{i:20}:{self.__dict__[i]}")        

    def load_dataLoader(self, 
            train_dataset: Dataset=None, 
            val_dataset: Dataset=None, 
            test_dataset: Dataset=None, 
            num_workers=0):
        ''' load DataLoader '''
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        if train_dataset:
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=num_workers)
        if val_dataset:
            self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=num_workers)
        if test_dataset:
            self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=num_workers)
    
    def load_model(self):
        ''' overload to load model '''
        self.model:nn.Module = None

    def load_optimizer(self):
        ''' overload to load optimizer '''
        self.optimizer:Optimizer = None

    def load_scheduler(self):
        ''' overload to load scheduler '''
        self.scheduler:lr_scheduler._LRScheduler = None

    def forward(self, data:torch.Tensor) ->torch.Tensor:
        ''' overload by your  forward function'''
        data = data.to(self.DEVICE)
        pred = self.model(data)
        return pred
        
    def loss(self, data:torch.Tensor, pred:torch.Tensor) ->torch.Tensor:
        ''' overload by your loss function'''
        loss = nn.L1Loss()(pred, data)
        return loss

    def train(self):
        if self.model==None or self.train_dataset==None:
            self.print('Have not load model or train_dataset')
            exit()
        else:
            self.load_optimizer()
            self.load_scheduler()
            os.makedirs(self.OUTPUT)
            self.mode = 'train'
            for file in os.listdir(self.WORKSPACE):
                if os.path.splitext(file)[-1] == '.py':
                    shutil.copy(file, os.path.join(self.OUTPUT,file))    
        ''' main code'''
        self.print(f"{'*' * 30} Start Train {'*' * 30}")
        num_train_batches = len(self.train_dataloader)
        batch_num_of_print = num_train_batches//self.TRAIN_PRING_NUM if num_train_batches//self.TRAIN_PRING_NUM else 1
        batch_num_of_val = num_train_batches//self.VAL_NUM if num_train_batches//self.VAL_NUM else 1
        batch_time = 0
        val_time = 0

        for epoch in range(self.START_EPOCH, self.EPOCH+1):
            self.epoch = epoch
            self.model.train()
            loss_tmp = []
            ''' print epoch info '''
            self.print(f'epoch: {epoch:>3d}/{self.EPOCH:>3d}')
            self.print(time.ctime())
            self.print(f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            self.print(f"{'*' * 40}")
            ''' train batch loop '''
            batch_t1 = time.time() 
            batch_t0 = batch_t1
            for batch, data in enumerate(self.train_dataloader): 
                self.batch = batch
                self.STEP = epoch+(batch+1)/num_train_batches
                ''' forward '''
                pred = self.forward(data)
                ''' loss '''
                loss = self.loss(data, pred)
                loss_tmp.append(loss.item())
                ''' backward '''
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ''' print batch info'''
                batch_t1 = time.time()
                batch_time = batch_time + batch_t1-batch_t0
                batch_t0 = batch_t1
                if batch % batch_num_of_print == (batch_num_of_print-1):
                    batch_time = batch_time/batch_num_of_print
                    need_time = (num_train_batches-(batch+1))*batch_time + (self.VAL_NUM-batch//batch_num_of_val)*val_time + (batch_time*num_train_batches + val_time*self.VAL_NUM)*(self.EPOCH - epoch)
                    need_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()+need_time))
                    average_loss = sum(loss_tmp)/len(loss_tmp)
                    self.print(f"Train Loss:{average_loss:>7f} [{batch+1:>6d}/{num_train_batches:>6d}] | Batch Time: {batch_time:>.2f}s | ETA: {need_time}")
                    with open(os.path.join(self.OUTPUT, self.TIMESTAMP + "-loss.csv"),'a') as f:
                        f.write(str(epoch+(batch+1)/num_train_batches) + ',' + str(average_loss))
                        f.write('\n')
                        loss_tmp = []
                    batch_time = 0

                ''' val '''   
                if self.VAL_NUM>0 and batch % batch_num_of_val == (batch_num_of_val-1):
                    self.print(f"Epoch:    [{epoch:>3d}/{self.EPOCH:>3d}]")
                    val_time = self.val(self.STEP)
                    self.model.train()

            ''' end of a epoch '''
            if self.scheduler:
                self.scheduler.step()

            ''' save checkpoint '''
            if self.CHECKPOINT_NUM>0:
                epoch_num_of_check = self.EPOCH//self.CHECKPOINT_NUM if self.EPOCH//self.CHECKPOINT_NUM else 1
                if (epoch-1) % epoch_num_of_check == (epoch_num_of_check-1):      
                    fileName = self.TIMESTAMP + '_' +str(epoch) +'.ckpt'
                    SavePath = os.path.join(self.OUTPUT, fileName)
                    self.print(f"model checkpoint SavePath: {SavePath}")
                    checkpoint = {'parameter': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}
                    if self.scheduler:
                        checkpoint['scheduler'] = self.scheduler.state_dict()
                    torch.save(checkpoint, SavePath)
            elif self.CHECKPOINT_NUM<0:                
                fileName = self.TIMESTAMP + '_' +str(epoch) +'.ckpt'
                SavePath = os.path.join(self.OUTPUT, fileName)
                self.print(f"model checkpoint SavePath: {SavePath}")
                checkpoint = {'parameter': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch}
                if self.scheduler:
                    checkpoint['scheduler'] = self.scheduler.state_dict()
                for file in os.listdir(self.OUTPUT):
                    if file.endswith('.ckpt'):
                        os.remove(os.path.join(self.OUTPUT, file))
                torch.save(checkpoint, SavePath)

        self.print(f"{'*' * 30} End Train {'*' * 30}")
        self.print(time.ctime())
        fileName = self.TIMESTAMP + '.pth'
        SavePath = os.path.join(self.OUTPUT, fileName)
        self.print(f"model SavePath: {SavePath}")
        torch.save(self.model.state_dict(), SavePath)
        return fileName
    
    def before_val(self):
        pass

    def evaluate(self, data, pred):
        ''' evaluate model in each val batch'''
        pass

    def after_val(self):
        pass
    
    def val(self, save_point=None):
        if self.model==None or self.val_dataset==None:
            self.print('Have not load model or val_dataset')
            exit()
        self.before_val()
        self.model.eval()
        t0 = time.time()
        num_val_batches = len(self.val_dataloader)
        val_loss = 0
        with torch.no_grad():
            for val_batch, data in enumerate(self.val_dataloader):
                if val_batch % (num_val_batches//2) == 0:
                    self.print(f"Valing:   [{val_batch:>6d}/{num_val_batches:>6d}]") 
                ''' forward '''
                pred = self.forward(data)
                loss = self.loss(data, pred)
                val_loss += loss.item()

                ''' evaluate '''
                self.evaluate(data, pred)
            
            ''' pring val info '''
            val_loss /= num_val_batches
            if save_point:
                with open(os.path.join(self.OUTPUT, self.TIMESTAMP + "-val_loss.csv"),'a') as f:
                    f.write(str(save_point) + ',' + str(val_loss))
                    f.write('\n')
                self.print_loss_figure()
        t1 = time.time()
        self.print(f"Val  Loss:{val_loss:>7f} | {len(self.val_dataset):>6d} | Val  Time: {t1 - t0:>.2f}s")
        self.after_val()
        return t1 - t0
    
    def before_test(self):
        pass

    def format(self, data, pred):
        ''' format pred '''
        pass

    def after_test(self):
        pass

    def test(self):
        if self.model==None or self.test_dataset==None:
            self.print('Have not load model or test_dataset')
            exit()
        self.before_test()
        self.model.eval()
        t0 = time.time()
        num_test_batches = len(self.test_dataloader)
        inference_time = 0
        with torch.no_grad():
            for test_batch, data in enumerate(self.val_dataloader):
                if test_batch % (num_test_batches//4) == 0:
                    self.print(f"Testing:   [{test_batch:>6d}/{num_test_batches:>6d}]") 
                ''' forward '''
                inference_time -= time.time()
                pred = self.forward(data)
                inference_time += time.time()
                ''' format '''
                self.format(data, pred)            
        t1 = time.time()
        self.print(f"Test  Time: {t1 - t0:>.2f}s | Average Inference Time: {inference_time/len(self.test_dataset):>.6f}")
        self.after_test()
        return t1 - t0
        
    def print_loss_figure(self):
        trainlossPath = os.path.join(self.OUTPUT, self.TIMESTAMP + "-loss.csv")
        vallossPath = os.path.join(self.OUTPUT, self.TIMESTAMP + "-val_loss.csv")
        with open(trainlossPath,'r',encoding='utf8') as f:
            txts = f.readlines()
            txts = [i.rstrip("\n") for i in txts]
            trainloss = np.array([i.split(',') for i in txts]).astype(np.float32)
        with open(vallossPath,'r',encoding='utf8') as f:
            txts = f.readlines()
            txts = [i.rstrip("\n") for i in txts]
            valloss = np.array([i.split(',') for i in txts]).astype(np.float32)
        plt.figure('loss')
        plt.plot(trainloss[:,0],trainloss[:,1],'r-',valloss[:,0],valloss[:,1],'b-')
        plt.ylim(0, (np.mean(trainloss[:,1])+np.mean(valloss[:,1]))/2+(np.std(trainloss[:,1])+np.std(valloss[:,1]))/2)
        plt.grid(visible=True)
        plt.title(self.TIMESTAMP)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(['trainloss','valloss'])
        plt.savefig(os.path.join(self.OUTPUT,self.TIMESTAMP+'.jpg'))

if __name__ == '__main__':
    pass
    
