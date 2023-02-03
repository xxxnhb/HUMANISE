import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'
# from model.pointtransformer2.pointtransformer2_semseg import pointtransformer2_feature_extractor as PointTransformer2Enc
#
# def _process_scene_model_static_dict(model_dict):
#     static_dict = {}
#     for key in model_dict.keys():
#         if 'enc_stages' in key or 'patch_embed' in key:
#             static_dict[key[7:]] = model_dict[key]
#             print(key[7:])
#     return static_dict
# # for k, v in loaded_dict.items():
# #     name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
# #     new_state_dict[name] = v #新字典的key值对应的value一一对应
# # loaded_model = torch.load('/mnt/disk_1/jinpeng/ptv2/exp/scannet/semseg-ptv2m2-0-base/model/model_best.pth')
print(torch.cuda.device_count())
# scene_model = PointTransformer2Enc(in_channels=9).cuda()
# # model_dict = torch.load('/mnt/disk_1/jinpeng/ptv2/exp/scannet/semseg-ptv2m2-0-base/model/model_best.pth')['state_dict']
# model_dict = torch.load('/mnt/disk_1/jinpeng/ptv2/exp/scannet/semseg-ptv2m2-0-base/model/model_dict_best.pth')
#
# # model_dict = torch.load('/mnt/disk_1/jinpeng/ptv2/exp/scannet/semseg-ptv2m2-0-base/model/model_dict_best.pth', map_location={'cuda:0':'cuda:1'})
# static_dict = _process_scene_model_static_dict(model_dict)
#
# scene_model.load_state_dict(static_dict)
# print('sss')