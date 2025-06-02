from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
from utils import to_device, postprocess_masks

import torchvision.transforms as transforms
from lora_image_encoder import LoRA_Sam
from mscan_encoder import Mscan_Encoder
import json
# from reverse_att2 import ReverseAttention, MaskRefineHead, RARefineModule

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")  # 指定测试过程的工作目录 默认值为workdir
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_ckpt", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")  # 加载sam或sammed模型文件
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")  # 使用bbox提示获取分割结果
    parser.add_argument("--point_num", type=int, default=1, help="point num")  # 指定点数
    parser.add_argument("--iter_point", type=int, default=1, help="iter num")  # 指定点提示的迭代次数
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    # parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")  # 如果使用sam-med2d的预训练权重，则设置为True
    parser.add_argument("--encoder_adapter", action="store_true", help="use adapter")  # 是否微调adapter层
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")  # 有固定的prompt文件吗 否则该值为none，它将在最新预测中自动生成
    # parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")  # 是否保存预测结果
    parser.add_argument("--save_pred", action="store_true", help="save result")  # 不写--save_pred就是False,写了就是True  

    ### CNN
    parser.add_argument('--mscan', type=str, default='tiny'),
    parser.add_argument('--mscan_ckpt', type=str, default='pretrain_model/mscan_t.pth'),
    # parser.add_argument('--mscan', type=str, default='large'),
    # parser.add_argument('--mscan_ckpt', type=str, default='pretrain_model/mscan_l.pth'),

    parser.add_argument('--image_size_cnn', type=int, default=256, help='image size used during training') 

    ### LoRA
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')

    ### val/test drop_rate=0
    parser.add_argument('--drop_rate', type=float, default=0, help='drop_rate')

    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def prompt_and_decoder_test(args, batched_input, MSCAN_model, SAM_model, image_embeddings, interm_embeddings, refiner=None):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = SAM_model.sam.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )
        
        ### 提取CNN特征
        image = torch.stack(
                [transforms.Resize(args.image_size_cnn)(img) for img in batched_input["image"]]
            )
        features = MSCAN_model(image)
        cnn_feature = features[4]

        low_res_masks, iou_predictions = SAM_model.sam.mask_decoder(
            cnn_feature=cnn_feature,
            image_embeddings = image_embeddings,
            image_pe = SAM_model.sam.prompt_encoder.get_dense_pe(),
            interm_embeddings=interm_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    # 如果提供了 refiner，则进行细化
    if refiner is not None:
        refined_mask = refiner(cnn_feature, low_res_masks)
        low_res_masks = refined_mask
    
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True

def load_model(checkpoint_path, args, device):
    """
    加载模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 实例化 LoRA_Sam 模型
    org_sam_model = sam_model_registry[args.model_type](args)
    SAM_model = LoRA_Sam(org_sam_model, args.rank).to(device)
    SAM_model.load_state_dict(checkpoint['sam_model_state_dict'], strict=False)

    MSCAN_model = Mscan_Encoder(args).to(device)
    MSCAN_model.load_state_dict(checkpoint['mscan_model_state_dict'])

    # refiner = RARefineModule(in_channels=256 if args.mscan=="tiny" else 512).to(args.device)
    # # 使用 features[2] (通道是64)，模块会自动适配到 256
    # refiner = RARefineModule(in_channels=256 if args.mscan=="tiny" else 512, hidden_channels=256 if args.mscan=="tiny" else 512).to(args.device)

    # refiner.load_state_dict(checkpoint['RA_refiner'])

    return SAM_model, MSCAN_model


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    SAM_model, MSCAN_model = load_model(args.lora_ckpt, args, args.device)
    SAM_model.eval()
    MSCAN_model.eval()
    # refiner.eval()

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }

        with torch.no_grad():
            image_encoder_output = SAM_model.sam.image_encoder(batched_input["image"])
            image_embeddings = image_encoder_output["vision_features"]
            interm_embeddings = image_encoder_output["interm_embeddings"]

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder_test(args, batched_input, MSCAN_model, SAM_model, image_embeddings, interm_embeddings, refiner=None)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
     
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder_test(args, batched_input, MSCAN_model, SAM_model, image_embeddings, interm_embeddings, refiner=None)
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        # ###
        # # print(masks) 
        # # 保存完整张量到文本文件
        # # with open("mask_output.txt", "w") as f:
        # #     f.write(str(masks.cpu()))
        # # 设置显示时的阈值，使其不截断
        # # torch.set_printoptions(threshold=float('inf'))
        # # mask_slice = slice(0, 1)
        # # # 然后再 print(masks)
        # # print(masks[:, mask_slice, :, :])

        # import matplotlib.pyplot as plt
        # # masks1 = torch.sigmoid(masks)
        # masks1 = masks.detach().cpu().numpy()  # 先 .cpu()，再 .numpy()
        # masks1 = masks1[0]  # 取出 batch 的第一个样本 => [N, H, W]
        # num_masks = masks1.shape[0]

        # plt.figure(figsize=(15, 5))

        # for i in range(num_masks):
        #     plt.subplot(1, num_masks, i + 1)
        #     plt.imshow(masks1[i], cmap='gray')
        #     plt.title(f"Mask {i}")
        #     plt.axis('off')

        # plt.tight_layout()
        # # plt.show()

        # for i in range(num_masks):
        #     plt.imsave(f"mask_{i}.png", masks1[i], cmap='gray')

        # ###

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
  
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir,f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
