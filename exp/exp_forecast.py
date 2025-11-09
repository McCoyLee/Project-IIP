import os
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

# === AMP 相关 ===
from torch.cuda.amp import autocast, GradScaler
torch.set_float32_matmul_precision("high")  # 可选：在 Ampere+/Ada 上略提速

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        # 记录 MoE 是否已在本进程中完成启用（convert_dense_ffn_to_moe）
        self._moe_converted = False

        # AMP GradScaler（DDP/DP/单卡通用）
        self.scaler = GradScaler(enabled=True)

    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            self.device = self.args.gpu

        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)

        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _align_target(self, outputs, batch_y):
        """
        将 batch_y 对齐到模型输出：取最后 pred_len 步，并按需选择目标通道。
        同时裁剪 outputs 到最后 pred_len 步（更稳健，兼容部分模型返回更长序列的情况）。
        """
        pred_len = self.args.output_token_len

        # 裁剪到最后 pred_len 步
        outputs = outputs[:, -pred_len:, :]
        batch_y = batch_y[:, -pred_len:, :]

        # 只评估目标通道
        if getattr(self.args, 'eval_target_only', False):
            tc = int(getattr(self.args, 'target_channel', -1))
            C = outputs.shape[-1]
            if tc < 0 or tc >= C:
                tc = C - 1
            outputs = outputs[..., tc:tc+1]
            batch_y = batch_y[..., tc:tc+1]

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # AMP 推理
                with autocast(dtype=torch.float16):
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

                    # 对齐标签与输出
                    outputs, batch_y = self._align_target(outputs, batch_y)

                    # 原有 covariate 逻辑（可选，仅在 args.covariate=True 时启用）
                    if self.args.covariate:
                        if self.args.last_token:
                            outputs = outputs[:, -self.args.output_token_len:, -1]
                            batch_y = batch_y[:, -self.args.output_token_len:, -1]
                        else:
                            outputs = outputs[:, :, -1]
                            batch_y = batch_y[:, :, -1]

                    loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)

        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            self.model.in_layer.train()
            self.model.out_layer.train()
        else:
            self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()

            # ===== [MoE Warmup Enable] 到 warmup 结束时再启用 MoE（若尚未启用） =====
            if getattr(self.args, 'use_moe', False) and hasattr(self.model, 'convert_dense_ffn_to_moe'):
                warmup_ep = getattr(self.args, 'moe_warmup_epochs', 5)
                if (epoch + 1) == warmup_ep and not self._moe_converted:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print(f"[MoE] enabling experts at epoch {epoch+1}")
                    try:
                        self.model.convert_dense_ffn_to_moe()
                        self._moe_converted = True
                    except Exception as e:
                        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                            print(f"[MoE] enable failed or already enabled: {e}")
            # ======================================================================

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # === AMP：前向与损失 ===
                with autocast(dtype=torch.float16):
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    if self.args.dp:
                        torch.cuda.synchronize()

                    # 对齐标签与输出（总是取最后 pred_len 步；可选目标通道）
                    outputs, batch_y = self._align_target(outputs, batch_y)

                    # 原有 covariate 逻辑（可选）
                    if self.args.covariate:
                        if self.args.last_token:
                            outputs = outputs[:, -self.args.output_token_len:, -1]
                            batch_y = batch_y[:, -self.args.output_token_len:, -1]
                        else:
                            outputs = outputs[:, :, -1]
                            batch_y = batch_y[:, :, -1]

                    loss = criterion(outputs, batch_y)

                    # ==== MoE aux：平均 + 系数 + warmup（warmup 期不加） ====
                    moe_terms = []
                    for m in self.model.modules():
                        if hasattr(m, "_moe_aux_total"):
                            aux = m._moe_aux_total
                            if not torch.is_tensor(aux):
                                aux = torch.tensor(float(aux), device=self.device)
                            else:
                                aux = aux.to(self.device)
                            if torch.isfinite(aux):
                                moe_terms.append(aux)
                    if moe_terms:
                        moe_aux = torch.stack(moe_terms).mean()  # 先对所有层取平均
                        coef = getattr(self.args, "moe_aux_coef", 0.1)
                        warm = getattr(self.args, "moe_warmup_epochs", 5)
                        if epoch + 1 > warm:  # warmup 期间不加入 aux
                            loss = loss + coef * moe_aux
                    # =====================================================

                # ===== NaN/Inf 保护：发现非有限 loss 就跳过该步 =====
                if not torch.isfinite(loss):
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("[warn] non-finite loss detected (NaN/Inf). Skip step.")
                    model_optim.zero_grad()
                    continue

                # === AMP：缩放反传 + 反缩放后裁剪 + step ===
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(model_optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(model_optim)
                self.scaler.update()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 逐段滚动到 test_pred_len
                out_len = self.args.output_token_len
                inference_steps = self.args.test_pred_len // out_len
                if self.args.test_pred_len % out_len != 0:
                    inference_steps += 1

                pred_segs = []
                for j in range(inference_steps):
                    if len(pred_segs) != 0:
                        # 1) 输入序列滚动 out_len 步
                        batch_x = torch.cat([batch_x[:, out_len:, :], pred_segs[-1]], dim=1)
                        # 2) 时间戳也同步滚动 out_len 步；从 y_mark 取第 j 段时间戳接到末尾
                        start = (j - 1) * out_len
                        end = min(j * out_len, batch_y_mark.size(1))
                        seg = batch_y_mark[:, start:end, :]
                        if seg.size(1) < out_len:
                            # 末段不够则用最后一个时间戳做简单填充
                            seg = torch.cat([seg, seg[:, -1:, :].repeat(1, out_len - seg.size(1), 1)], dim=1)
                        batch_x_mark = torch.cat([batch_x_mark[:, out_len:, :], seg], dim=1)

                    # AMP 推理
                    with autocast(dtype=torch.float16):
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    pred_segs.append(outputs[:, -out_len:, :])

                # 拼接所有段，并裁成 test_pred_len
                pred_y = torch.cat(pred_segs, dim=1)[:, :self.args.test_pred_len, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # 可视化（若开启），通道选择与评测保持一致
                if self.args.visualize and i % 2 == 0:
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    # 选择可视化通道：若只评目标通道，则按 target_channel；否则默认最后一列
                    c = int(getattr(self.args, 'target_channel', -1)) if getattr(self.args, 'eval_target_only', False) else -1
                    C = pred.shape[-1]
                    if c < 0 or c >= C:
                        c = C - 1
                    gt = np.array(true[0, :, c])
                    pd = np.array(pred[0, :, c])
                    visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))

                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()

        preds = torch.cat(preds, dim=0).numpy()   # [N, H, C]
        trues = torch.cat(trues, dim=0).numpy()   # [N, H, C]
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)

        # ============ 只评目标通道（例如 OT） ============
        if getattr(self.args, 'eval_target_only', False):
            c = int(getattr(self.args, 'target_channel', -1))
            C = preds.shape[-1]
            if c < 0 or c >= C:
                c = C - 1
            preds = preds[..., c:c+1]
            trues = trues[..., c:c+1]
            print(f"[Eval] only target channel idx={c}; preds={preds.shape}, trues={trues.shape}")
        # ==============================================

        # 原有 covariate 逻辑（若启用，则再取最后一列）
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]

        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return
