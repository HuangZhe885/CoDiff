

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import torch.nn as nn
# from icecream import ic
# from opencood.models.sub_modules.pillar_vfe import PillarVFE
# from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
# from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
# from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
# from opencood.models.sub_modules.downsample_conv import DownsampleConv
# from opencood.models.sub_modules.naive_compress import NaiveCompressor, NaiveReCompressor
# from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
# from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
# from opencood.utils.transformation_utils import normalize_pairwise_tfm
# from opencood.models.fuse_modules.diffusion_fuse_predictnoise import DifussionFusion


# class NoiseHead(nn.Module):
#     def __init__(self, in_channels=256):
#         super(NoiseHead, self).__init__()
        
#         # 定义卷积层
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1),  # 这里in_channels可以根据输入特征的C确定
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 进行池化，减小H和W维度
#         )
        
#         # 全局平均池化
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, C, 1, 1)
        
#         # 全连接层，将池化后的特征映射到标量
#         self.fc = nn.Sequential(
#             nn.Flatten(),  # 将 (B, C, 1, 1) 转换为 (B, C)
#             nn.Linear(128, 1),  # 128是卷积之后的通道数
#             nn.Sigmoid()  # 映射到 0-1 之间
#         )

#     def forward(self, x):
#         x = self.conv(x)  # 卷积层处理
#         x = self.global_avg_pool(x)  # 全局平均池化
#         x = self.fc(x)  # 全连接层输出标量
#         return x


# class PointPillarBaselineMultiscale(nn.Module):
#     """
#     F-Cooper implementation with point pillar backbone.
#     """
#     def __init__(self, args):
#         super(PointPillarBaselineMultiscale, self).__init__()

#         self.pillar_vfe = PillarVFE(args['pillar_vfe'],
#                                     num_point_features=4,
#                                     voxel_size=args['voxel_size'],
#                                     point_cloud_range=args['lidar_range'])
#         self.scatter = PointPillarScatter(args['point_pillar_scatter'])
#         is_resnet = args['base_bev_backbone'].get("resnet", True) # default true
#         if is_resnet:
#             self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
#         else:
#             self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
#         self.voxel_size = args['voxel_size']

#         self.fusion_net = nn.ModuleList()
#         for i in range(len(args['base_bev_backbone']['layer_nums'])):
#             if args['fusion_method'] == "max":
#                 self.fusion_net.append(MaxFusion())
#             if args['fusion_method'] == "att":
#                 self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
#         self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

#         self.diffusion = DifussionFusion(args)  # 后续需要把配置写进配置文件里
#         self.noise_head = NoiseHead()


#         self.shrink_flag = False
#         if 'shrink_header' in args:
#             self.shrink_flag = True
#             self.shrink_conv = DownsampleConv(args['shrink_header'])
#             self.out_channel = args['shrink_header']['dim'][-1]

#         self.compression = False
#         if "compression" in args:
#             self.compression = True
#             self.naive_compressor = NaiveCompressor(256, args['compression'])     #diar 256
#             self.naive_re_compressor = NaiveReCompressor(256, args['compression'])    #diar 256
#             self.spatial_compressor = NaiveCompressor(64, args['spatial_compression'])   #diar 64 

#         self.cross_attention = ScaledDotProductAttention(256)  #diar 256

#         self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
#                                   kernel_size=1)
#         self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
#                                   kernel_size=1)
#         self.use_dir = False
#         if 'dir_args' in args.keys():
#             self.use_dir = True
#             self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
#                                   kernel_size=1) # BIN_NUM = 2
 
#         if 'backbone_fix' in args.keys() and args['backbone_fix']:
#             self.backbone_fix()

#         if 'train_mode' in args.keys():
#             self.train_mode = args['train_mode']
#         else:
#             self.train_mode = 'finetune'

#     def backbone_fix(self):
#         """
#         Fix the parameters of backbone during finetune on timedelay。
#         """
#         for p in self.pillar_vfe.parameters():
#             p.requires_grad = False

#         for p in self.scatter.parameters():
#             p.requires_grad = False

#         for p in self.backbone.parameters():
#             p.requires_grad = False

#         if self.compression:
#             for p in self.naive_compressor.parameters():
#                 p.requires_grad = False
#         if self.shrink_flag:
#             for p in self.shrink_conv.parameters():
#                 p.requires_grad = False

#         for p in self.cls_head.parameters():
#             p.requires_grad = False
#         for p in self.reg_head.parameters():
#             p.requires_grad = False

#     def forward(self, data_dict):
#         voxel_features = data_dict['processed_lidar']['voxel_features']
#         voxel_coords = data_dict['processed_lidar']['voxel_coords']
#         voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
#         record_len = data_dict['record_len']

#         batch_dict = {'voxel_features': voxel_features,
#                       'voxel_coords': voxel_coords,
#                       'voxel_num_points': voxel_num_points,
#                       'record_len': record_len}
#         # n, 4 -> n, c
#         batch_dict = self.pillar_vfe(batch_dict)
#         # n, c -> N, C, H, W
#         batch_dict = self.scatter(batch_dict)
#         # calculate pairwise affine transformation matrix
#         _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
#         normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

#         spatial_features = batch_dict['spatial_features']


#         # multiscale fusion
#         feature_list = self.backbone.get_multiscale_feature(spatial_features)
#         fused_feature_list = []
#         for i, fuse_module in enumerate(self.fusion_net):
#             fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix))
#         fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 


#         if self.shrink_flag:
#             fused_feature = self.shrink_conv(fused_feature)


#         if self.train_mode == 'coalign':

#             copy_fused_feature = fused_feature.clone()
#             if self.compression:
#                 copy_fused_feature = self.naive_compressor(copy_fused_feature)
#                 fused_feature_diff = self.naive_re_compressor(copy_fused_feature)
            
#             # fused_feature = (fused_feature + fused_feature_diff)/2
#             # print("fused_feature",fused_feature.shape)

#             # 基于cross attention的融合
#             # B, C, H, W = fused_feature.shape
#             # fused_feature = fused_feature.view(B, C, -1).permute(2, 0, 1)
#             # fused_feature_diff = fused_feature_diff.view(B, C, -1).permute(2, 0, 1)
#             # fused_feature = self.cross_attention(fused_feature, fused_feature_diff, fused_feature_diff)
#             # fused_feature = fused_feature.permute(1, 2, 0).view(B, C, H, W)
 
#             psm = self.cls_head(fused_feature)
#             rm = self.reg_head(fused_feature)
#             noise = self.noise_head(fused_feature)

#             output_dict = {'cls_preds': psm,
#                         'reg_preds': rm,
#                         'feature': fused_feature,
#                         'diff_feature': fused_feature_diff,
#                         'noise': noise}

#             if self.use_dir:
#                 output_dict.update({'dir_preds': self.dir_head(fused_feature)})

#             return output_dict
        
#         elif self.train_mode == 'diffusion':
#             # copy_fused_feature = fused_feature.clone()
#             noise = self.noise_head(fused_feature)
#             if self.compression:
#                 fused_feature = self.naive_compressor(fused_feature)
#                 spatial_features = self.spatial_compressor(spatial_features) 
#             diff_loss = self.diffusion(fused_feature, spatial_features, record_len, normalized_affine_matrix, noise)
#             return diff_loss
        
#         else: # finetune 模式
#             copy_fused_feature = fused_feature.clone()
#             noise = self.noise_head(fused_feature)
#             #压缩特征, 压缩到8
#             if self.compression:
#                 copy_fused_feature = self.naive_compressor(copy_fused_feature)   # [ batch , 8, 100, 252]
#                 spatial_features = self.spatial_compressor(spatial_features)     # [ batched cavs, 8, 200, 504]

#             fused_feature_diff = self.diffusion.sample(copy_fused_feature, spatial_features, record_len, normalized_affine_matrix, noise)
#             # fused_feature_diff: [B, 8, 100, 252]  是经过压缩的 而fused_feature是未经过压缩的 [B, 256, 100, 252]
#             if self.compression:   
#                 fused_feature_diff = self.naive_re_compressor(fused_feature_diff)
            
#             # 手动设置的权重融合
#             final_fused_feature = fused_feature*0.5 + fused_feature_diff*0.5
                       
                    
#             psm = self.cls_head(final_fused_feature)
#             rm = self.reg_head(final_fused_feature)

#             output_dict = {'cls_preds': psm,
#                         'reg_preds': rm,
#                         # 'fused_feature': fused_feature,
#                         # 'diff_feature': fused_feature_diff,
#                         # 'diffusion_loss': diff_loss
#                         }

#             if self.use_dir:
#                 output_dict.update({'dir_preds': self.dir_head(final_fused_feature)})

#             return output_dict



# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor, NaiveReCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.diffusion_fuse import DifussionFusion

class PointPillarBaselineMultiscale(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarBaselineMultiscale, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True) # default true
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.diffusion = DifussionFusion(args)  # 后续需要把配置写进配置文件里

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])     #diar 256
            self.naive_re_compressor = NaiveReCompressor(256, args['compression'])    #diar 256
            self.spatial_compressor = NaiveCompressor(64, args['spatial_compression'])   #diar 64 

        self.cross_attention = ScaledDotProductAttention(256)  #diar 256

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

        if 'train_mode' in args.keys():
            self.train_mode = args['train_mode']
        else:
            self.train_mode = 'finetune'

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']


        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 


        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)


        if self.train_mode == 'coalign':

            copy_fused_feature = fused_feature.clone()
            if self.compression:
                copy_fused_feature = self.naive_compressor(copy_fused_feature)
                fused_feature_diff = self.naive_re_compressor(copy_fused_feature)
            
            
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            output_dict = {'cls_preds': psm,
                        'reg_preds': rm,
                        'feature': fused_feature,
                        'diff_feature': fused_feature_diff}

            if self.use_dir:
                output_dict.update({'dir_preds': self.dir_head(fused_feature)})

            return output_dict
        
        elif self.train_mode == 'diffusion':
            # copy_fused_feature = fused_feature.clone()
            if self.compression:
                fused_feature = self.naive_compressor(fused_feature)
                spatial_features = self.spatial_compressor(spatial_features) 
            diff_loss = self.diffusion(fused_feature, spatial_features, record_len, normalized_affine_matrix)
            return diff_loss
        
        else: # finetune 模式
            # ''' diffusion部分
            # 深拷贝复制一份fused_feature,用于填充邻居特征
            copy_fused_feature = fused_feature.clone()

            #压缩特征, 压缩到8
            if self.compression:
                copy_fused_feature = self.naive_compressor(copy_fused_feature)   # [ batch , 8, 100, 252]
                spatial_features = self.spatial_compressor(spatial_features)     # [ batched cavs, 8, 200, 504]


            diff_loss = self.diffusion(copy_fused_feature, spatial_features, record_len, normalized_affine_matrix)
            # 下面是diffusion采样部分
            #  "copy_fused_feature"  ([4, 8, 100, 252])
            fused_feature_diff = self.diffusion.sample(copy_fused_feature, spatial_features, record_len, normalized_affine_matrix)
            # fused_feature_diff: [B, 8, 100, 252]  是经过压缩的 而fused_feature是未经过压缩的 [B, 256, 100, 252]
            if self.compression:   
                fused_feature_diff = self.naive_re_compressor(fused_feature_diff)
            # print("fused_feature",fused_feature.shape)
            # print("fused_feature_diff",fused_feature_diff.shape)
            # 手动设置的权重融合
            final_fused_feature = fused_feature*0.5 + fused_feature_diff*0.5
                       
                    
            psm = self.cls_head(final_fused_feature)
            rm = self.reg_head(final_fused_feature)

            output_dict = {'cls_preds': psm,
                        'reg_preds': rm,
                        # 'fused_feature': fused_feature,
                        # 'diff_feature': fused_feature_diff,
                        # 'diffusion_loss': diff_loss
                        }

            if self.use_dir:
                output_dict.update({'dir_preds': self.dir_head(final_fused_feature)})

            return output_dict