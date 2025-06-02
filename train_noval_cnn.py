from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched, ValidationDataset
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
# from apex import amp    # 混合精度训练需要安装apex
import random

from utils import to_device, split_param_groups, postprocess_masks, save_masks
import torchvision.transforms as transforms
from lora_image_encoder import LoRA_Sam
from mscan_encoder import Mscan_Encoder
import json
# from reverse_att2 import ReverseAttention, MaskRefineHead, RARefineModule
from utils import DiceLoss, sample_data


def parse_args():
    parser = argparse.ArgumentParser()
    ### Training
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")  
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")  
    parser.add_argument("--data_path_train", type=str, default="asps_data/train", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_ckpt", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")  
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")  # 交互式分割次数
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")  
    # parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")  # 是否微调adapter层
    parser.add_argument("--encoder_adapter", action="store_true", help="use adapter")  # 写了就是True,不写就是False
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")   # 是否需要混合精度训练

    parser.add_argument('--iterations', type=int, default=400*20, help='the number of iterations for training')    # 400*40
    parser.add_argument('--save_iter', default=400) # 5000  原来400


    ### CNN
    parser.add_argument('--mscan', type=str, default='tiny'),
    parser.add_argument('--mscan_ckpt', type=str, default='pretrain_model/mscan_t.pth'),
    parser.add_argument('--image_size_cnn', type=int, default=256, help='image size used during training') 
    parser.add_argument("--lr_cnn", type=float, default=1e-5, help="learning rate")

    ### LoRA
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    # parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')

    ### Validation
    parser.add_argument("--data_path_val", type=str, default="asps_data/val", help="val data path") 
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--patience", type=int, default=8, help="Patience for early stopping")
    parser.add_argument('--drop_rate', type=float, default=0, help='drop_rate')
    # parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")  ### bool("False")=True
    parser.add_argument("--save_pred", action="store_true", help="save result")  # 不写--save_pred就是False,写了就是True

    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


###
def prompt_and_decoder_train(args, batched_input, MSCAN_model, SAM_model, image_embeddings, interm_embeddings, decoder_iter = False):
    # 提取点提示
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    # 当处于解码器迭代模式时（decoder_iter=True），对 Prompt Encoder 的计算图进行自动梯度关闭
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = SAM_model.prompt_encoder(   
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = SAM_model.prompt_encoder(
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

    cnn_feature_mid = features[2]   ### 若mask_num不为1,也需要repeat操作

    B, _, _, _ =  cnn_feature.shape
    cnn_feature_repeat = []
    for i in range(B):
        cnn_feature_temp = cnn_feature[i]
        cnn_feature_temp = cnn_feature_temp.repeat(args.mask_num, 1, 1, 1)
        cnn_feature_repeat.append(cnn_feature_temp)
    
    cnn_feature = torch.cat(cnn_feature_repeat, dim=0)
    # print(cnn_feature.shape)

    # # 使用提示编码器的输出以及图像嵌入进行分割预测，输出低分辨率的分割掩码和每个掩码的 IoU 预测值
    low_res_masks, iou_predictions = SAM_model.mask_decoder(
        cnn_feature = cnn_feature,
        image_embeddings = image_embeddings,
        image_pe = SAM_model.prompt_encoder.get_dense_pe(),  # 从提示编码器提取的密集位置编码
        interm_embeddings=interm_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
    # 如果启用了多掩码模式（args.multimask=True），保留 IoU 最高的掩码
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)  # IoU 最高的掩码被筛选并保存在 low_res_masks 中
    # 将低分辨率掩码上采样到与输入图像一致的分辨率
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False)
    # print(masks)
    return masks, low_res_masks, iou_predictions, cnn_feature


# 将普通 DataLoader 包装成无限迭代器
def infinite_loader(dataset, batch_size, num_workers, pin_memory=True, shuffle=True):
    base_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    while True:
        for batch in base_loader:
            yield batch


def train_fixed_iters(args,
                      train_dataset,
                      MSCAN_model,
                      SAM_model,
                      optimizer_cnn,
                      optimizer,
                      criterion):

    # 1) 准备 checkpoint 目录
    checkpoint_dir = os.path.join(os.path.join(f"{args.work_dir}/models", args.run_name))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to {checkpoint_dir}")

    # 2) 无限 DataLoader
    loader = infinite_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    # 3) 模式设定
    SAM_model.train()     # SAM 的 image_encoder 冻结
    MSCAN_model.train()  # MSCAN 需要训练

    # 4) 固定迭代次数
    pbar = tqdm(range(1, args.iterations + 1), desc="Total iters")

    for itr in pbar:
        # 4.1 获取一批数据并预处理
        batched = next(loader)
        batched = stack_dict_batched(batched)
        batched = to_device(batched, args.device)

        # 随机选 boxes 或 points
        if random.random() > 0.5:
            batched["point_coords"] = None
            flag = "boxes"
        else:
            batched["boxes"] = None
            flag = "point"

        # 冻结 SAM 除 LoRA 以外参数
        for name, param in SAM_model.image_encoder.named_parameters():
            if "linear_a" in name or "linear_b" in name or "Adapter" in name:
                param.requires_grad = True   
            else:
                param.requires_grad = False

        # 4.2 第一轮前向（非交互）—— 生成初始掩码
        labels = batched["label"]
        # SAM 编码
        img = batched["image"]
        image_encoder_out = SAM_model.image_encoder(img)
        image_embeddings = image_encoder_out["vision_features"]
        interm_embeddings = image_encoder_out["interm_embeddings"]

        # repeat emb & interm
        B, _, _, _ = image_embeddings.shape
        emb_rep = [image_embeddings[i].repeat(args.mask_num,1,1,1) for i in range(B)]
        image_embeddings = torch.cat(emb_rep, dim=0)

        _, B1, _, _, _ = interm_embeddings.shape
        interm_rep = [interm_embeddings[:,i].unsqueeze(1).repeat(1,args.mask_num,1,1,1)
                      for i in range(B1)]
        interm_embeddings = torch.cat(interm_rep, dim=1)

        # 调用 prompt + decoder
        masks, low_res_masks, iou_preds, cnn_feat = prompt_and_decoder_train(
            args, batched, MSCAN_model, SAM_model,
            image_embeddings, interm_embeddings, decoder_iter=False
        )

        # loss & backward for first pass
        loss_first = criterion(masks, labels, iou_preds)
        loss_first.backward()

        optimizer.step()
        optimizer_cnn.step()
        optimizer.zero_grad()
        optimizer_cnn.zero_grad()

        if itr % 50 == 0:
            print(f'[Iter {itr}], first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

        # 4.3 8 次交互式迭代
        #   每次迭代，冻结 SAM image_encoder，优化其他模块
        #   并根据上次预测生成新的点提示，用于下一次解码
        image_embeddings = image_embeddings.detach().clone()
        interm_embeddings = interm_embeddings.detach().clone()

        # for name, param in SAM_model.named_parameters():
        #     param.requires_grad = not name.startswith("sam.image_encoder")
        
        ### 在第一次迭代后,冻结image_encoder中的参数,专注于优化其他模块的参数(此时LoRA的参数也会被冻结)
        for name, param in SAM_model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = False   
            else:
                param.requires_grad = True

        masks_iter = masks
        low_res_iter = low_res_masks
        iou_iter = iou_preds

        init_mask_num = np.random.randint(1, args.iter_point - 1)

        for iter_id in range(args.iter_point):
            # 随机在某次迭代取消所有提示，模拟无提示
            if iter_id == init_mask_num or iter_id == args.iter_point-1:
                batched = setting_prompt_none(batched)

            # 前向，只计算 prompt_encoder，没有梯度
            masks_iter, low_res_iter, iou_iter, cnn_feat = prompt_and_decoder_train(
                args, batched, MSCAN_model, SAM_model,
                image_embeddings, interm_embeddings, decoder_iter=True
            )

            # 计算迭代损失并保持计算图
            loss_iter = criterion(masks_iter, labels, iou_iter)
            loss_iter.backward(retain_graph=True)

            optimizer.step()
            optimizer_cnn.step()
            optimizer.zero_grad()
            optimizer_cnn.zero_grad()

            # 下次迭代生成新的提示（除了最后一次）
            if iter_id != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched = generate_point(
                    masks_iter, labels, low_res_iter, batched, point_num
                )
                batched = to_device(batched, args.device)

            # 每50 steps 打印一次中间指标
            if itr % 50 == 0:
                if iter_id == init_mask_num or iter_id == args.iter_point - 1:
                    print(f'[Iter {itr}] iter_id {iter_id+1}/{args.iter_point}, mask prompt, metrics: {SegMetrics(masks_iter, labels, args.metrics)}')
                else:
                    print(f'[Iter {itr}] iter_id {iter_id+1}/{args.iter_point}, point {point_num} prompt, metrics: {SegMetrics(masks_iter, labels, args.metrics)}')

            
        param_names_to_save_batch=[]        
    
        ### 查看哪些参数可训练
        for name, param in SAM_model.named_parameters():
            if param.requires_grad:
                param_names_to_save_batch.append(name)
                # print(name)
            elif "linear_a" in name or "linear_b" in name:
                param_names_to_save_batch.append(name)
                # print(name)   

        # 4.4 定期保存
        if itr % args.save_iter == 0 or itr == args.iterations:
            fname = f"{str(itr).zfill(7)}.pth"
            path = os.path.join(checkpoint_dir, fname)
            checkpoint = {
                'sam_model_state_dict': {k: v for k, v in SAM_model.state_dict().items() if k in param_names_to_save_batch},
                'mscan_model_state_dict': MSCAN_model.state_dict(),
            }

            torch.save(checkpoint, path)
            print(f"[Iter {itr}] checkpoint saved: {path}")

        pbar.set_postfix(loss=loss_iter.item())


def main(args):
    ### 构造SAM_model
    org_sam_model = sam_model_registry[args.model_type](args).to(args.device) 
    SAM_model = org_sam_model
    SAM_model.cuda()

    param_names_to_save=[]        
    
    ### 查看可训练参数
    for name, param in SAM_model.named_parameters():
        if param.requires_grad:
            param_names_to_save.append(name)
            # print(name)
    

    ### 构造MSCAN_model
    MSCAN_model = Mscan_Encoder(args)
    MSCAN_model.cuda()
    
    # for name, param in MSCAN_model.named_parameters():
    #     if param.requires_grad:
            # print(name)

    ### 分别设置优化器
    optimizer = optim.AdamW(split_param_groups(SAM_model), lr=args.lr)
    optimizer_cnn = optim.AdamW(MSCAN_model.parameters(), lr=args.lr_cnn, weight_decay=1e-4)
    # optimizer_cnn = optim.AdamW(list(MSCAN_model.parameters()) + list(refiner.parameters()), lr=args.lr_cnn, weight_decay=1e-4)

    ### Loss计算
    criterion = FocalDiceloss_IoULoss()

    # 学习率调度器
    if args.lr_scheduler:
        # 为 optimizer 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
        print('*******Use MultiStepLR for optimizer')

        # 为 optimizer_cnn 设置学习率调度器 (可以和 optimizer 使用相同的策略，也可以不同)
        scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[5, 10, 15, 20], gamma=0.5)
        print('*******Use MultiStepLR for optimizer_cnn')

    # 加载检查点
    # if args.resume is not None:
    #     with open(args.resume, "rb") as f:
    #         checkpoint = torch.load(f)
    #         model.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
    #         print(f"*******load {args.resume}")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    ### 加载训练集
    train_dataset = TrainingDataset(args.data_path_train, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))   

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))


    # for epoch in range(0, args.epochs):
    #     SAM_model.train()
    #     MSCAN_model.train()
    #     # refiner.train()

    #     train_metrics = {}
    #     start = time.time()
    #     os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
    #     train_losses, train_iter_metrics = train_one_epoch(args, MSCAN_model, SAM_model, optimizer_cnn, optimizer, train_loader, epoch, criterion)
        
    #     scheduler = args.lr_scheduler
    #     scheduler_cnn = args.lr_scheduler
    #     if scheduler is not None:
    #         scheduler.step()
    #     if scheduler_cnn is not None:
    #         scheduler_cnn.step()

    #     train_iter_metrics = [metric / len(train_loader) for metric in train_iter_metrics]
    #     train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

    #     average_loss = np.mean(train_losses)

    #     lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
    #     lr_cnn = scheduler_cnn.get_last_lr()[0] if scheduler_cnn is not None else args.lr_cnn

    #     # loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")
    #     loggers.info(f"epoch: {epoch + 1}, lr: {lr}, lr_cnn: {lr_cnn}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

    #     end = time.time()
    #     print("Run epoch time: %.2fs" % (end - start))
    # 不再按 epoch 分，而是按固定迭代次数总训练
    
    train_fixed_iters(
        args=args,
        train_dataset=train_dataset,
        MSCAN_model=MSCAN_model,
        SAM_model=SAM_model,
        optimizer_cnn=optimizer_cnn,
        optimizer=optimizer,
        criterion=criterion
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)

