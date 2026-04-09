"""
单模态与解耦有效性实验 (Table 4)
目的：证明特征解耦(FD)能去除冗余，提升单模态学习效果

实验设计：
1. 仅用语言(L)、仅用视觉(V)、仅用音频(A)进行训练
2. 对比：不加FD的单模态准确率 vs 加入FD的单模态准确率

实现：直接复用原始DMD的模型结构，不做任何简化
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config_regression
from data_loader import MMDataLoader
from utils import assign_gpu, setup_seed
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.DMD import MSE
from trains.utils import MetricsTop, dict_to_str
from trains.subNets.transformers_encoder.transformer import TransformerEncoder


class SingleModalityModel(nn.Module):
    """
    单模态模型 - 完全复用DMD的架构，不做任何简化
    """
    def __init__(self, args, modality='l', use_fd=True):
        super(SingleModalityModel, self).__init__()
        self.modality = modality
        self.use_fd = use_fd
        self.args = args

        # 获取维度
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.d = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.output_dropout = args.output_dropout
        self.attn_mask = args.attn_mask

        # 设置输入维度和卷积核大小（与DMD一致）
        if modality == 'l':
            self.orig_d = args.feature_dims[0]
            self.len = 50
            self.conv_kernel_size = args.conv1d_kernel_size_l
        elif modality == 'v':
            self.orig_d = args.feature_dims[2]
            self.len = 50
            self.conv_kernel_size = args.conv1d_kernel_size_v
        else:  # 'a'
            self.orig_d = args.feature_dims[1]
            self.len = 50
            self.conv_kernel_size = args.conv1d_kernel_size_a

        # 1. 投影层（与DMD一致）
        self.proj = nn.Conv1d(self.orig_d, self.d, kernel_size=self.conv_kernel_size, padding=0, bias=False)

        # 2. 特征解耦层（与DMD一致）
        self.encoder_s = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_c = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)

        # 3. 解码器（与DMD一致）
        self.decoder = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)

        # 计算序列长度
        self.seq_len = self.len - self.conv_kernel_size + 1
        combined_dim_low = self.d

        # 4. 用于cosine sim的投影（与DMD的proj_cosine_*一致）
        self.proj_cosine = nn.Linear(combined_dim_low * self.seq_len, combined_dim_low)

        # 5. 用于align c的投影（与DMD的align_c_*一致）
        self.align_c = nn.Linear(combined_dim_low * self.seq_len, combined_dim_low)

        # 6. Self-attention for c（与DMD的self_attentions_c_*一致）
        self.self_attentions_c = self.get_network(self_type=modality)

        # 7. fc layers for homogeneous graph distillation（与DMD的proj1_*_low等一致）
        self.proj1_low = nn.Linear(combined_dim_low * self.seq_len, combined_dim_low)
        self.proj2_low = nn.Linear(combined_dim_low, combined_dim_low * self.seq_len)
        self.out_layer_low = nn.Linear(combined_dim_low * self.seq_len, 1)

    def get_network(self, self_type='l', layers=-1):
        """与DMD的get_network完全一致"""
        if self_type == 'l':
            embed_dim, attn_dropout = self.d, self.attn_dropout
        elif self_type == 'a':
            embed_dim, attn_dropout = self.d, self.attn_dropout
        elif self_type == 'v':
            embed_dim, attn_dropout = self.d, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x, return_all=False):
        """
        前向传播 - 完全遵循DMD的forward逻辑
        """
        batch_size = x.size(0)

        # 转置为 [batch, feature_dim, seq_len]
        x = x.transpose(1, 2)

        # 投影（与DMD一致）
        proj_x = self.proj(x)  # [batch, d, seq_len']

        # 特征解耦（与DMD一致）
        s = self.encoder_s(proj_x)  # [batch, d, seq_len']
        c = self.encoder_c(proj_x)  # [batch, d, seq_len']

        # 重建（与DMD一致）
        recon = self.decoder(torch.cat([s, c], dim=1))  # [batch, d, seq_len']

        # 对重建后的特征再次编码（与DMD一致，用于cycle consistency loss）
        s_r = self.encoder_s(recon)  # [batch, d, seq_len']

        # Pool features for orthogonality loss（与DMD一致，在permute之前）
        s_pooled = s.mean(dim=2)  # [batch, d]
        c_pooled = c.mean(dim=2)  # [batch, d]

        # Permute（与DMD一致）
        s = s.permute(2, 0, 1)  # [seq_len', batch, d]
        c = c.permute(2, 0, 1)  # [seq_len', batch, d]

        # 准备HomoGD的输入（与DMD一致）
        # c.transpose(0, 1).contiguous().view(batch_size, -1)
        c_for_homo = c.transpose(0, 1).contiguous().view(batch_size, -1)  # [batch, d * seq_len']

        if not self.use_fd:
            # 不使用FD：使用原始投影特征
            features_for_homo = proj_x.transpose(1, 2).contiguous().view(batch_size, -1)
        else:
            # 使用FD：使用c特征（与DMD的HomoGD一致）
            features_for_homo = c_for_homo

        # HomoGD分类头（与DMD完全一致）
        repr_low = self.proj1_low(features_for_homo)  # [batch, d]
        hs_proj_low = self.proj2_low(
            F.dropout(F.relu(repr_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_low += features_for_homo  # 残差连接（与DMD一致）
        logits_low = self.out_layer_low(hs_proj_low)  # [batch, 1]

        if return_all:
            # 计算各种中间特征（与DMD一致）

            # proj_cosine（与DMD一致）
            s_for_cosine = s.transpose(0, 1).contiguous().view(batch_size, -1)
            proj_s = self.proj_cosine(s_for_cosine)  # [batch, d]

            # align_c（与DMD一致）
            c_sim = self.align_c(c_for_homo)  # [batch, d]

            # Self-attention for c（与DMD一致）
            c_att = self.self_attentions_c(c)
            if type(c_att) == tuple:
                c_att = c_att[0]
            c_att = c_att[-1]  # [batch, d]

            return {
                'output': logits_low,
                'logits_low': logits_low,
                'repr_low': repr_low,
                's': s,  # [seq_len', batch, d]
                'c': c,  # [seq_len', batch, d]
                'recon': recon,  # [batch, d, seq_len']
                's_r': s_r,  # [batch, d, seq_len']
                'origin': proj_x,  # [batch, d, seq_len']
                's_pooled': s_pooled,  # [batch, d]
                'c_pooled': c_pooled,  # [batch, d]
                'c_sim': c_sim,  # [batch, d]
                'proj_s': proj_s,  # [batch, d]
                'c_att': c_att,  # [batch, d]
            }
        else:
            return {'output': logits_low}


class SingleModalityTrainer:
    """单模态训练器 - 完全复用DMD的训练逻辑"""
    def __init__(self, args, modality='l', use_fd=True):
        self.args = args
        self.modality = modality
        self.use_fd = use_fd
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader):
        """训练模型 - 完全复用DMD的训练逻辑"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.args.patience, verbose=True
        )

        epochs = 0
        best_valid = 1e8
        best_epoch = 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'

        while True:
            epochs += 1
            model.train()
            train_loss = 0.0
            y_pred, y_true = [], []

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # 获取对应模态的数据
                    if self.modality == 'l':
                        x = batch_data['text'].to(self.args.device)
                    elif self.modality == 'v':
                        x = batch_data['vision'].to(self.args.device)
                    else:  # 'a'
                        x = batch_data['audio'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    optimizer.zero_grad()
                    output = model(x, return_all=True)

                    # 任务损失（与DMD一致）
                    loss_task = self.criterion(output['logits_low'], labels)

                    if self.use_fd:
                        # 重建损失（与DMD一致：loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])）
                        loss_recon = self.MSE(output['recon'], output['origin'])

                        # cycle consistency loss（与DMD一致：loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])）
                        # 注意：DMD中s的shape是[seq_len, batch, d]，需要permute
                        loss_s_sr = self.MSE(
                            output['s'].permute(1, 2, 0),  # [batch, d, seq_len]
                            output['s_r']  # [batch, d, seq_len]
                        )

                        # 正交损失（与DMD一致）
                        target_ort = torch.full((output['s_pooled'].size(0),), -1).to(self.args.device)
                        loss_ort = self.cosine(output['s_pooled'], output['c_pooled'], target_ort)

                        # margin loss（与DMD一致）
                        c_sim = output['c_sim']
                        ids, feats = [], []
                        for i in range(labels.size(0)):
                            feats.append(c_sim[i].view(1, -1))
                            ids.append(labels[i].view(1, -1))
                        feats = torch.cat(feats, dim=0)
                        ids = torch.cat(ids, dim=0)
                        loss_sim = self.sim_loss(ids, feats)

                        # 总损失（与DMD完全一致：(loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1）
                        loss = loss_task + (loss_s_sr + loss_recon + (loss_sim + loss_ort) * 0.1) * 0.1
                    else:
                        # 不使用FD时，只使用任务损失
                        loss = loss_task

                    loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)

                    optimizer.step()
                    train_loss += loss.item()

                    y_pred.append(output['output'].cpu())
                    y_true.append(labels.cpu())

            train_loss /= len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            print(f"Epoch {epochs}: Train Loss={train_loss:.4f}, "
                  f"MAE={train_results['MAE']:.4f}, Corr={train_results['Corr']:.4f}")

            # 验证
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            print(f"Validation: MAE={val_results['MAE']:.4f}, Corr={val_results['Corr']:.4f}")

            scheduler.step(val_results['Loss'])

            # 保存最佳模型
            if val_results['Loss'] < best_valid:
                best_valid = val_results['Loss']
                best_epoch = epochs
                save_path = f"./pt/single_{self.modality}_{'fd' if self.use_fd else 'nofd'}.pth"
                torch.save(model.state_dict(), save_path)

            # 早停
            if epochs - best_epoch >= self.args.early_stop:
                print(f"Early stopping at epoch {epochs}")
                break

        # 加载最佳模型
        model.load_state_dict(torch.load(save_path))
        return model

    def do_test(self, model, dataloader, mode="TEST"):
        """测试模型"""
        model.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    if self.modality == 'l':
                        x = batch_data['text'].to(self.args.device)
                    elif self.modality == 'v':
                        x = batch_data['vision'].to(self.args.device)
                    else:  # 'a'
                        x = batch_data['audio'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    output = model(x, return_all=False)
                    y_pred.append(output['output'].cpu())
                    y_true.append(labels.cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        results['Loss'] = self.criterion(pred, true).item()

        return results


def run_single_modality_experiment(dataset_name='mosei', seeds=[1111, 1112, 1113, 1114, 1115]):
    """
    运行单模态实验
    对比：不加FD vs 加入FD
    """
    config_file = Path(__file__).parent / "config" / "config.json"
    args = get_config_regression('dmd', dataset_name, config_file)

    args['device'] = assign_gpu([0])
    args['train_mode'] = 'regression'

    # 创建结果保存目录
    os.makedirs("./pt", exist_ok=True)
    os.makedirs("./result", exist_ok=True)

    results = {}

    for modality in ['l', 'v', 'a']:
        modality_name = {'l': 'Language', 'v': 'Vision', 'a': 'Audio'}[modality]
        print(f"\n{'='*60}")
        print(f"Running experiments for {modality_name} modality")
        print(f"{'='*60}")

        results[modality] = {'no_fd': [], 'with_fd': []}

        for use_fd in [False, True]:
            fd_name = "with FD" if use_fd else "without FD"
            print(f"\n--- {fd_name} ---")

            seed_results = []
            for seed in seeds:
                print(f"\nSeed: {seed}")
                setup_seed(seed)

                # 创建数据加载器
                dataloader = MMDataLoader(args, num_workers=4)

                # 创建模型
                model = SingleModalityModel(args, modality=modality, use_fd=use_fd)
                model = model.cuda()

                # 训练
                trainer = SingleModalityTrainer(args, modality=modality, use_fd=use_fd)
                model = trainer.do_train(model, dataloader)

                # 测试
                test_results = trainer.do_test(model, dataloader['test'])
                print(f"Test Results: MAE={test_results['MAE']:.4f}, "
                      f"Corr={test_results['Corr']:.4f}, "
                      f"Acc_7={test_results.get('Acc_7', 0):.4f}")

                seed_results.append(test_results)

            # 计算平均值
            avg_results = {}
            for key in ['MAE', 'Corr', 'Acc_7', 'Acc_2', 'F1']:
                if key in seed_results[0]:
                    values = [r[key] for r in seed_results]
                    avg_results[key] = np.mean(values)
                    avg_results[key + '_std'] = np.std(values)

            key = 'with_fd' if use_fd else 'no_fd'
            results[modality][key] = avg_results

    # 打印结果表格
    print("\n" + "="*80)
    print("Table 4: Single Modality Experiment Results")
    print("="*80)
    print(f"{'Modality':<15} {'FD':<10} {'MAE':<12} {'Corr':<12} {'Acc_7':<12}")
    print("-"*80)

    for modality in ['l', 'v', 'a']:
        modality_name = {'l': 'Language', 'v': 'Vision', 'a': 'Audio'}[modality]

        # 不加FD
        r = results[modality]['no_fd']
        print(f"{modality_name:<15} {'No':<10} "
              f"{r.get('MAE', 0):.4f}±{r.get('MAE_std', 0):.4f}   "
              f"{r.get('Corr', 0):.4f}±{r.get('Corr_std', 0):.4f}   "
              f"{r.get('Acc_7', 0):.4f}±{r.get('Acc_7_std', 0):.4f}")

        # 加FD
        r = results[modality]['with_fd']
        print(f"{'':<15} {'Yes':<10} "
              f"{r.get('MAE', 0):.4f}±{r.get('MAE_std', 0):.4f}   "
              f"{r.get('Corr', 0):.4f}±{r.get('Corr_std', 0):.4f}   "
              f"{r.get('Acc_7', 0):.4f}±{r.get('Acc_7_std', 0):.4f}")
        print()

    # 保存结果
    with open("./result/single_modality_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to ./result/single_modality_results.json")

    return results


if __name__ == "__main__":
    # 运行实验
    results = run_single_modality_experiment(dataset_name='mosei', seeds=[1111])
