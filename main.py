import torch  # 引入 PyTorch
from dataset import get_data_transforms  # 從 dataset.py 載入資料轉換函式
from torchvision.datasets import ImageFolder  # 用於影像資料夾的資料集
import numpy as np  # 數值計算套件
import random  # 亂數控制
import os  # 檔案系統操作
from torch.utils.data import DataLoader  # PyTorch 的資料載入器
from dataset import MVTecDataset  # MVTec 資料集類別
import torch.backends.cudnn as cudnn  # CUDA cuDNN 加速
import argparse  # 命令列參數處理
from test import evaluation, visualization, test  # 測試、評估與可視化函式
from torch.nn import functional as F  # 引入 PyTorch 的函式介面
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork  # 假設你的 DRAEM 定義在 models/draem.py


def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋


# 蒸餾損失函數
def distillation_loss(teacher_features, student_features):
    cos_loss = torch.nn.CosineSimilarity()
    if not isinstance(teacher_features, (list, tuple)):
        teacher_features, student_features = [teacher_features
                                              ], [student_features]

    loss = 0
    for i in range(len(teacher_features)):
        loss += torch.mean(1 - cos_loss(
            teacher_features[i].view(teacher_features[i].shape[0], -1),
            student_features[i].view(student_features[i].shape[0], -1)))
    return loss


def train(_arch_, _class_, epochs, save_pth_path):
    # 訓練流程
    print(f"🔧 類別: {_class_} | Epochs: {epochs}")
    learning_rate = 0.005  # 學習率
    # batch_size = 16  # 批次大小
    batch_size = 8  # 批次大小
    image_size = 256  # 輸入影像大小

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 選擇裝置
    print(f"🖥️ 使用裝置: {device}")

    # 資料轉換
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = f'./mvtec/{_class_}/train'  # 訓練資料路徑
    test_path = f'./mvtec/{_class_}'  # 測試資料路徑

    # 載入訓練與測試資料
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path,
                             transform=data_transform,
                             gt_transform=gt_transform,
                             phase="test")

    # 建立 DataLoader
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)

    # 教師模型（預訓練 DRAEM，凍結）
    teacher_encoder = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    teacher_decoder = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    # === Step 2: 載入 checkpoint ===
    encoder_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle_.pckl",
        map_location=device,
        weights_only=True)
    decoder_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle__seg.pckl",
        map_location=device,
        weights_only=True)
    teacher_encoder.load_state_dict(encoder_ckpt)
    teacher_decoder.load_state_dict(decoder_ckpt)
    # 重要：載入權重後再移到設備
    teacher_encoder = teacher_encoder.to(device)
    teacher_decoder = teacher_decoder.to(device)
    teacher_encoder.eval()
    teacher_decoder.eval()

    # 學生模型（需要訓練）
    student_encoder = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    student_decoder = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    student_encoder = student_encoder.to(device)
    student_decoder = student_decoder.to(device)

    # 建立優化器，訓練學生模型
    optimizer = torch.optim.Adam(list(student_encoder.parameters()) +
                                 list(student_decoder.parameters()),
                                 lr=learning_rate,
                                 betas=(0.5, 0.999))

    # 建立輸出資料夾
    save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    os.makedirs(save_pth_dir, exist_ok=True)

    # 設定最佳權重檔案存放路徑
    best_ckp_path = os.path.join(save_pth_dir, f'best_{_arch_}_{_class_}.pth')

    # 初始化最佳分數
    best_score = -1

    # 訓練迴圈
    for epoch in range(epochs):
        student_encoder.train()
        student_decoder.train()
        loss_list = []

        for img, label in train_dataloader:
            img = img.to(device)

            # 教師模型推理
            with torch.no_grad():
                teacher_recon = teacher_encoder(img)
                teacher_input = torch.cat([img, teacher_recon], dim=1)
                teacher_seg = teacher_decoder(teacher_input)

            # 學生模型推理
            student_recon = student_encoder(img)
            student_input = torch.cat([img, student_recon], dim=1)
            student_seg = student_decoder(student_input)

            # 蒸餾損失：比較相同語義的輸出
            recon_loss = distillation_loss(teacher_recon, student_recon)
            seg_loss = distillation_loss(teacher_seg, student_seg)

            total_loss = recon_loss + seg_loss

            optimizer.zero_grad()
            total_loss.backward()  # 修正：使用 total_loss
            optimizer.step()
            loss_list.append(total_loss.item())

        print(
            f"📘 Epoch [{epoch + 1}/{epochs}] | Loss: {np.mean(loss_list):.4f}")

        # 每個 epoch 都進行一次評估（使用學生模型）
        auroc_px, auroc_sp, aupro_px = evaluation(student_encoder,
                                                  student_decoder,
                                                  test_dataloader, device)
        print(f"🔍 評估 | Pixel AUROC: {auroc_px:.3f}")

        # 如果表現更好則儲存學生模型
        if auroc_px > best_score:
            best_score = auroc_px
            torch.save(
                {
                    'encoder': student_encoder.state_dict(),
                    'decoder': student_decoder.state_dict()
                }, best_ckp_path)
            print(f"💾 更新最佳模型 → {best_ckp_path}")

    # 訓練結束回傳最佳結果
    return best_ckp_path, best_score, auroc_sp, aupro_px, student_encoder, student_decoder


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os
    import torch

    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    args = parser.parse_args()

    setup_seed(111)  # 固定隨機種子
    save_visual_path = f"results/{args.arch}_{args.category}"
    save_pth_path = f"pths/best_{args.arch}_{args.category}"
    # 開始訓練，並接收最佳模型路徑與結果
    best_ckp, auroc_px, auroc_sp, aupro_px, bn, decoder = train(
        args.arch, args.category, args.epochs, save_pth_path)

    print(f"最佳模型: {best_ckp}")

    # 存訓練指標到 CSV
    df_metrics = pd.DataFrame([{
        'Category': args.category,
        'Pixel_AUROC': auroc_px,
        'Sample_AUROC': auroc_sp,
        'Pixel_AUPRO': aupro_px,
        'Epochs': args.epochs
    }])
    metrics_name = f"metrics_{args.arch}_{args.category}.csv"
    df_metrics.to_csv(metrics_name,
                      mode='a',
                      header=not os.path.exists(metrics_name),
                      index=False)

    # 🔥 訓練結束後自動產生可視化結果
    visualization(args.arch,
                  args.category,
                  ckp_path=best_ckp,
                  save_path=save_visual_path)
