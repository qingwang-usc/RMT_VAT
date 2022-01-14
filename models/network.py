# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:29:18 2019

@author: mwa
"""

from torch import nn
from models.densenet3d import densenet59
from models.densenet import densenet121, densenet169, densenet201, densenet161
import torch.nn.functional as F
from models.layers import SaveFeatures, UnetBlock_, UnetBlock, UnetBlock3d_, UnetBlock3d
import torch

def ComputePara(net):
    params = list(net.parameters())
    k = 0
#    if not os.path.exists(savePath):
#         file_w = open(savePath,'w')
#    file_w = open(savePath,'r+')  
#    file_w.read()
    for i in params:
        l = 1
#        print("layer structure:" + str(list(i.size())))
#        file_w.write("layer structure:" + str(list(i.size())) + '\n') 
        for j in i.size():
            l *= j
#        print("layer paramenters:"+str(l))
#        file_w.write("layer paramenters:" + str(l) + '\n')
        k += l
    print("network paramenters:"+str(k))
#    file_w.write("network paramenters:" + str(k) + '\n') 
#    file_w.close()

def x2d_to_volumes(x):
    n,c,h,w,d = x.shape
    x_start = x[:,:,:,:,0:1]
    x_end = x[:,:,:,:,d-1:d]
    x = torch.cat((x_start,x,x_end),4)
    x_3d = x[:,0,:,:,0:3].permute(0,3,1,2)
    for i in range(1,d):
        x_tmp = x[:,0,:,:,i:i+3].permute(0,3,1,2)
        x_3d = torch.cat((x_3d,x_tmp),0)
    return x_3d

def dim_tran(x):      
    x = x.permute(1,2,3,0)
    x = x.unsqueeze(0)
    return x
    
class DenseUnet_2d(nn.Module):

    def __init__(self, densenet='densenet161'):
        super().__init__()

        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        # self.up1 = UnetBlock_(2208,2112,768)
        # self.up2 = UnetBlock(768,384,768)
        # self.up3 = UnetBlock(384,96, 384)
        # self.up4 = UnetBlock(96,96, 96)

        self.up1 = UnetBlock_(2208, 2112, 768)
        self.up2 = UnetBlock(768, 384)
        self.up3 = UnetBlock(384, 96)
        self.up4 = UnetBlock(96, 96)


        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        # self.deconv = nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)


        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)


        # x_fea = self.deconv(x)
        x_fea = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_fea = self.conv1(x_fea)
        if dropout:
            x_fea = F.dropout2d(x_fea, p=0.3)
        x_fea = F.relu(self.bn1(x_fea))
        x_out = self.conv2(x_fea)

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove()
        
class hybridnet(nn.Module):

    def __init__(self, densenet='densenet59'):
        super().__init__()
        
        self.denseunet_2d = DenseUnet_2d()
        model_path = './models_save/DenseUnet_2d/30.pkl'
        model_dict = torch.load(model_path)
        model_dict_clone = model_dict.copy()
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]
        self.denseunet_2d.load_state_dict(model_dict,False)
       
        base_model = densenet59
        layers = list(base_model().children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        self.up1 = UnetBlock3d_(504,496,504)
        self.up2 = UnetBlock3d(504,224,224)
        self.up3 = UnetBlock3d(224,192,192)
        self.up4 = UnetBlock3d(192,96,96)

        self.bn1 = nn.BatchNorm3d(64) 
        self.bn2 = nn.BatchNorm3d(64) 
        self.conv1 = nn.Conv3d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 3, kernel_size=1, padding=0)

    def forward(self, input):       
        n,c,h,w,d = input.shape
        x_volumes = x2d_to_volumes(input)
        out_2d,fea_2d = self.denseunet_2d(x_volumes)
        out_3d,fea_3d = dim_tran(out_2d)*250,dim_tran(fea_2d)
        x_3d = torch.cat((input,out_3d),1)
        
        x_3d = F.relu(self.rn(x_3d))
        x_3d = self.up1(x_3d, self.sfs[3].features)
        x_3d = self.up2(x_3d, self.sfs[2].features)
        x_3d = self.up3(x_3d, self.sfs[1].features)
        x_3d = self.up4(x_3d, self.sfs[0].features)

        x_out = F.upsample(x_3d, size=(h,w,d), mode='trilinear')
        x_out = self.conv1(x_out)
        x_out = F.dropout3d(x_out,p=0.3)
        x_out = F.relu(self.bn1(x_out))
        
        x_out = x_out + fea_3d
        x_out = self.conv2(x_out)
        x_out_dropout = F.dropout3d(x_out,p=0.1)
        x_out_bn = F.relu(self.bn2(x_out_dropout))
        final_result = self.conv3(x_out_bn)
        
        return final_result

    def close(self):
        for sf in self.sfs: sf.remove()  