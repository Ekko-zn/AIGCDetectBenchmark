import torch
import torch.nn as nn
from networks.resnet_PSM import resnet50
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F
import numpy as np


class SA_layer(nn.Module):
    def __init__(self, dim=128, head_size=4):
        super(SA_layer, self).__init__()
        self.mha=nn.MultiheadAttention(dim, head_size)
        self.ln1=nn.LayerNorm(dim)
        self.fc1=nn.Linear(dim, dim)
        self.ac=nn.ReLU()
        self.fc2=nn.Linear(dim, dim)
        self.ln2=nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim=x.shape
        x=torch.transpose(x,1,0)
        y,_=self.mha(x,x,x)
        x=self.ln1(x+y)
        x=torch.transpose(x,1,0)
        x=x.reshape(batch_size*len_size, fea_dim)
        x=x+self.fc2(self.ac(self.fc1(x)))
        x=x.reshape(batch_size,len_size, fea_dim)
        x=self.ln2(x)
        return x


class COOI(): # Coordinates On Original Image
    def __init__(self):
        self.stride=32
        self.cropped_size=224
        self.score_filter_size_list=[[3,3],[2,2]]
        self.score_filter_num_list=[3,3]
        self.score_nms_size_list=[[3,3],[3,3]]
        self.score_nms_padding_list=[[1,1],[1,1]]
        self.score_corresponding_patch_size_list=[[224, 224], [112, 112]]
        self.score_filter_type_size=len(self.score_filter_size_list)

    def get_coordinates(self, fm, scale):
        with torch.no_grad():
            batch_size, _, fm_height, fm_width=fm.size()
            scale_min=torch.min(scale, axis=1, keepdim=True)[0].long()
            scale_base=(scale-scale_min).long()//2 #torch.div(scale-scale_min,2,rounding_mode='floor')
            input_loc_list=[]
            for type_no in range(self.score_filter_type_size):
                score_avg=nn.functional.avg_pool2d(fm, self.score_filter_size_list[type_no], stride=1) #(7,2048,5,5), (7,2048,6,6) 这里做了一个特征池化，应该就是计算了每个patch的分数
                score_sum=torch.sum(score_avg, dim=1, keepdim=True) #(7,1,5,5), (7,1,6,6) #since the last operation in layer 4 of the resnet50 is relu, thus the score_sum are greater than zero
                _,_,score_height,score_width=score_sum.size()
                patch_height, patch_width=self.score_corresponding_patch_size_list[type_no]

                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat=score_sum.view(batch_size, -1)
                    value_max,loc_max_flat=torch.max(score_sum_flat, dim=1)
                    #loc_max=torch.stack((torch.div(loc_max_flat,score_width,rounding_mode='floor'), loc_max_flat%score_width), dim=1)
                    loc_max=torch.stack((loc_max_flat//score_width, loc_max_flat%score_width), dim=1)
                    # 这里是NMS
                    top_patch=nn.functional.max_pool2d(score_sum, self.score_nms_size_list[type_no], stride=1, padding=self.score_nms_padding_list[type_no])
                    value_max=value_max.view(-1,1,1,1)
                    erase=(top_patch!=value_max).float() # due to relu operation, the value are greater than 0, thus can be erase by multiply by 1.0/0.0
                    score_sum=score_sum*erase

                    # location in the original images
                    loc_rate_h=(2*loc_max[:,0]+fm_height-score_height+1)/(2*fm_height)
                    loc_rate_w=(2*loc_max[:,1]+fm_width-score_width+1)/(2*fm_width)
                    loc_rate=torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center=(scale_base+scale_min*loc_rate).long()
                    loc_top=loc_center[:,0]-patch_height//2
                    loc_bot=loc_center[:,0]+patch_height//2+patch_height%2
                    loc_lef=loc_center[:,1]-patch_width//2
                    loc_rig=loc_center[:,1]+patch_width//2+patch_width%2
                    loc_tl=torch.stack((loc_top, loc_lef), dim=1)
                    loc_br=torch.stack((loc_bot, loc_rig), dim=1)

                    # For boundary conditions
                    loc_below=loc_tl.detach().clone() # too low
                    loc_below[loc_below>0]=0
                    loc_br-=loc_below
                    loc_tl-=loc_below
                    loc_over=loc_br-scale.long() # too high
                    loc_over[loc_over<0]=0
                    loc_tl-=loc_over
                    loc_br-=loc_over
                    loc_tl[loc_tl<0]=0 # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor=torch.stack(input_loc_list, dim=1) # (7,6,4)
            #print(input_loc_tensor)
            return input_loc_tensor

class Patch5Model(nn.Module):
    # 这是网络结构
    def __init__(self):
        super(Patch5Model, self).__init__()
        self.resnet = resnet50(pretrained=True) #debug
        self.COOI=COOI()
        self.mha_list=nn.Sequential(
                        SA_layer(128, 4),
                        SA_layer(128, 4),
                        SA_layer(128, 4)
                      )  # 创建一个包含三个自注意力层（Self-Attention Layer）的神经网络模型。每个自注意力层都有128个输入特征和4个注意力头
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1=nn.Linear(2048, 128)
        self.ac=nn.ReLU()
        self.fc=nn.Linear(128,1)

    def forward(self, input_img, cropped_img, scale):
        
        x = cropped_img

        batch_size, p, _, _ =x.shape #[batch_size, 3, 224, 224]

        fm, whole_embedding=self.resnet(x) # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        #print(whole_embedding.shape)
        #print(fm.shape)
        s_whole_embedding=self.ac(self.fc1(whole_embedding))#128， 全局信息
        s_whole_embedding=s_whole_embedding.view(-1, 1, 128)
        #print(s_whole_embedding.shape)

        input_loc=self.COOI.get_coordinates(fm.detach(), scale) # 获取坐标

        _,proposal_size,_=input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fm.device)  # [N, 4, 3, 224, 224]
        
        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t,l,b,r=input_loc[batch_no, proposal_no]
                #print('************************')
                img_patch=input_img[batch_no][:, t:b, l:r]
                #print(img_patch.size())
                _, patch_height, patch_width=img_patch.size()
                if patch_height==224 and patch_width==224:
                    window_imgs[batch_no, proposal_no]=img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no+1]=F.interpolate(img_patch[None,...], size=(224, 224),
                                                            mode='bilinear',
                                                            align_corners=True)  # [N, 4, 3, 224, 224]
        #print(window_imgs.shape)
        #exit()

        # 根据坐标获取到了patch图像，然后使用resnet网络获得到patch的特征
        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224)  # [N*4, 3, 224, 224] 
        _, window_embeddings=self.resnet(window_imgs.detach()) #[batchsize*self.proposalN, 2048]
        s_window_embedding=self.ac(self.fc1(window_embeddings))#[batchsize*self.proposalN, 128]
        s_window_embedding=s_window_embedding.view(-1, proposal_size, 128)
        #print(s_window_embedding.shape)
        #exit()

        # 将patch和原始图像的特征在第一个维度上进行拼接
        all_embeddings=torch.cat((s_window_embedding, s_whole_embedding), 1)#[1, 1+self.proposalN, 128]
        #all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        #print(all_embeddings.shape)
        
        # mha_list是构造的多头注意力层，对获取到的特征进行一个融合
        all_embeddings=self.mha_list(all_embeddings)
        #print(all_embeddings.shape)
        all_logits=self.fc(all_embeddings[:,-1])
        #exit()
        
        return all_logits


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model=Patch5Model()

       

        if not self.isTrain or opt.continue_train:
            #self.model = resnet50(num_classes=1)
            self.model=Patch5Model()

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        # 处理没有GPU的情况
        if len(opt.gpu_ids)==0:
            self.model.to('cpu')
        else:
            self.model.cuda()


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input_img = data[0] # (batch_size, 6, 3, 224, 224)，输入图像
        self.cropped_img = data[1].to(self.device) # 进行中心剪裁过的图像，在自定义的dataloader中会返回这些数据
        self.label = data[2].to(self.device).float() # (batch_size)
        self.scale = data[3].to(self.device).float()
        #self.imgname = data[4]
    def forward(self):
        self.output = self.model(self.input_img, self.cropped_img, self.scale)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

