import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_forecast import Exp_Forecast

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timer-XL')

    parser.add_argument('--input_token_stride', type=int, default=None,
                        help='time patch stride; if None, defaults to input_token_len')
    parser.add_argument('--ci_backbone', action='store_true', default=False,
                        help='share encoder across variables (Channel-Independent)')
    parser.add_argument('--head_type', type=str, default='last',
                        choices=['last','mean','tokenwise'],
                        help='prediction head: last/mean for pred_len, tokenwise for per-token outputs')

    # ---- AdaRevIN / De-Stationary 开关（新增） ----
    parser.add_argument('--use_adanorm', action='store_true', default=False,
                        help='enable AdaRevIN (adaptive reversible instance normalization)')
    parser.add_argument('--adanorm_alpha', type=str, default='per_channel',
                        choices=['none','scalar','per_channel','mlp'],
                        help='alpha type for input-side mixing')
    parser.add_argument('--adanorm_beta', type=str, default='scalar',
                        choices=['scalar','per_channel','none'],
                        help='beta type for output-side mixing')
    parser.add_argument('--adanorm_use_ema', action='store_true', default=True,
                        help='use EMA detrending inside AdaRevIN')
    parser.add_argument('--ema_gamma', type=float, default=0.995,
                        help='EMA gamma for detrending')
    parser.add_argument('--use_desta', action='store_true', default=False,
                        help='enable De-Stationary Attention (logits modulation)')

    # === UniCA：融合阶段（前置/后置）与稳定性超参 ===
    parser.add_argument('--unica_stage', type=str, default='post',
                        choices=['pre', 'post'],
                        help='apply UniCA before encoder (pre) or after encoder (post)')
    parser.add_argument('--unica_res_scale', type=float, default=0.05,
                        help='residual scale when fusion=res_add')
    parser.add_argument('--unica_gamma_scale', type=float, default=0.1,
                        help='gamma perturb scale when fusion=film_gate')
    parser.add_argument('--unica_beta_scale', type=float, default=0.05,
                        help='beta shift scale when fusion=film_gate')
    parser.add_argument('--unica_dropout', type=float, default=0.0,
                        help='dropout on UniCA condition branch')
    parser.add_argument('--unica_smooth_gate_ks', type=int, default=3,
                        help='temporal smoothing kernel for gate (odd)')
    parser.add_argument('--unica_smooth_beta_ks', type=int, default=3,
                        help='temporal smoothing kernel for beta (odd)')
    parser.add_argument('--unica_init_gate_bias', type=float, default=-2.0,
                        help='initial bias for gate (small gate at start)')
    parser.add_argument('--unica_init_alpha_bias', type=float, default=-2.0,
                        help='initial bias for global mixing alpha (small at start)')

    # ---- eval options (新增) ----
    parser.add_argument('--eval_target_only', action='store_true',
                        help='only evaluate the target channel (e.g., OT)')
    parser.add_argument('--target_channel', type=int, default=6,
                        help='0-based index of target channel; ETTm1 OT=6')

    # ---- RevIN ----
    parser.add_argument('--revin', action='store_true', default=False)
    parser.add_argument('--revin_affine', action='store_true', default=True)
    parser.add_argument('--revin_eps', type=float, default=1e-5)

    # ---- UniCA 开关 ----
    parser.add_argument('--use_unica', action='store_true', default=False,
                        help='enable UniCA-style covariate homogenization + fusion')
    parser.add_argument('--unica_bottleneck', type=int, default=128,
                        help='bottleneck dim of covariate adapter')
    parser.add_argument('--unica_fusion', type=str, default='film_gate',
                        choices=['film_gate', 'res_add'],
                        help='fusion mode: film_gate or res_add')
    parser.add_argument('--unica_exclude_target', action='store_true', default=False,
                        help='exclude target channel statistics when building covariate features')

    # ---- MoE ----
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--moe_init_noise', type=float, default=0.0)
    parser.add_argument('--moe_topk', type=int, default=1, help='Top-k experts for routing (1 or 2).')
    parser.add_argument('--moe_capacity_factor', type=float, default=1.25, help='capacity factor for each expert.')
    parser.add_argument('--moe_gate_temp', type=float, default=1.0, help='softmax temperature for gating.')
    parser.add_argument('--moe_gate_noise_std', type=float, default=0.0, help='std of Gaussian noise added to gate logits (training only).')
    parser.add_argument('--moe_lb_alpha', type=float, default=0.02, help='weight for load-balance loss (importance & load) ala Switch.')
    parser.add_argument('--moe_imp_alpha', type=float, default=0.0, help='(optional) extra weight for importance loss.')
    parser.add_argument('--moe_zloss_beta', type=float, default=0.0, help='z-loss weight on router logits (stabilize).')
    parser.add_argument('--moe_entropy_reg', type=float, default=0.0, help='entropy regularization on router probs.')
    parser.add_argument('--moe_learnable_temp', action='store_true', default=False)
    parser.add_argument('--moe_gate_dropout', type=float, default=0.0)
    parser.add_argument('--moe_kl_alpha', type=float, default=0.0)

    # ---- AFS-Gate（新增）----
    parser.add_argument('--use_afs_gate', action='store_true', default=False,
                        help='enable adaptive frequency selection gate')
    parser.add_argument('--afs_bands', type=int, default=8,
                        help='number of learnable sub-bands in AFS')
    parser.add_argument('--afs_res_scale', type=float, default=0.2,
                        help='residual scale when fusing AFS output')
    parser.add_argument('--afs_init_gamma', type=float, default=0.0,
                        help='init gamma=0 -> identity frequency mask')
    parser.add_argument('--afs_place', type=str, default='block_post',
                        choices=['block_pre', 'block_post'],
                        help='apply AFS before or after encoder block')
    parser.add_argument('--afs_as_moe_prior', action='store_true', default=False,
                        help='expose AFS alpha as MoE prior (requires gate support)')

    # ---- basic config ----
    parser.add_argument('--task_name', type=str, required=True, default='forecast', help='task name, options:[forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='timer_xl', help='model name, options: [timer_xl, timer, moirai, moment]')
    parser.add_argument('--seed', type=int, default=2021, help='seed')

    # ---- data loader ----
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--test_flag', type=str, default='T', help='test domain')

    # ---- forecasting task ----
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--input_token_len', type=int, default=24, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='output token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')

    # ---- model define ----
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--e_layers', type=int, default=1, help='encoder layers')
    parser.add_argument('--d_model', type=int, default=512, help='d model')
    parser.add_argument('--n_heads', type=int, default=8, help='n heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='d ff')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--covariate', action='store_true', help='use cov', default=False)
    parser.add_argument('--node_num', type=int, default=100, help='number of nodes')
    parser.add_argument('--node_list', type=str, default='23,37,40', help='number of nodes for a tree')
    parser.add_argument('--use_norm', action='store_true', help='use norm', default=False)
    parser.add_argument('--nonautoregressive', action='store_true', help='nonautoregressive', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    parser.add_argument('--output_attention', action='store_true', help='output attention', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--flash_attention', action='store_true', help='flash attention', default=False)

    # ---- Linear-branch 融合（新增） ----
    parser.add_argument('--use_linear_branch', action='store_true', default=False)
    parser.add_argument('--linear_res_scale', type=float, default=0.2)
    parser.add_argument('--linear_init_gate', type=float, default=-2.0)

    # ---- adaptation ----
    parser.add_argument('--adaptation', action='store_true', help='adaptation', default=False)
    parser.add_argument('--pretrain_model_path', type=str, default='pretrain_model.pth', help='pretrain model path')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='few shot ratio')

    # ---- optimization ----
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--valid_last', action='store_true', help='valid last', default=False)
    parser.add_argument('--last_token', action='store_true', help='last token', default=False)

    # ---- GPU ----
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--ddp', action='store_true', help='Distributed Data Parallel', default=False)
    parser.add_argument('--dp', action='store_true', help='Data Parallel', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # ---- LLM-based model ----
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--stride', type=int, default=8)

    # ---- TTM ----
    parser.add_argument("--n_vars", type=int, default=7, help='number of variables')
    parser.add_argument("--factor", type=int, default=2, help='expansion factor of hidden layer')
    parser.add_argument("--mode", type=str, default="mix_channel", help="allowed values: common_channel, mix_channel")
    parser.add_argument("--AP_levels", type=int, default=0, help="number of attention patching levels")
    parser.add_argument("--use_decoder", action="store_true", help="use decoder", default=True)
    parser.add_argument("--d_mode", type=str, default="common_channel", help="allowed values: common_channel, mix_channel")
    parser.add_argument("--layers", type=int, default=8, help="number of layers in ttm")
    parser.add_argument("--hidden_dim", type=int, default=16, help="hidden dimension in ttm")

    # ---- Time-LLM ----
    parser.add_argument("--ts_vocab_size", type=int, default=1000, help="size of a small collection of text prototypes in llm")
    parser.add_argument("--domain_des", type=str, default="The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.", help="domain description")
    parser.add_argument("--llm_model", type=str, default="LLAMA", help="LLM model, LLAMA, GPT2, BERT, OPT are supported")
    parser.add_argument("--llm_layers", type=int, default=6, help="number of layers in llm")

    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.patch_size is not None and args.patch_size > 0:
        args.input_token_len = args.patch_size
    if args.stride is not None and args.stride > 0:
        args.input_token_stride = args.stride
    if args.input_token_stride is None:
        args.input_token_stride = args.input_token_len

    if args.head_type == 'tokenwise' and args.input_token_stride != args.input_token_len:
        print("[Warn] tokenwise 头要求 stride==patch_len；已自动改为 head_type='last'")
        args.head_type = 'last'

    args.node_list = [int(x) for x in args.node_list.split(',')]

    if args.dp:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    elif args.ddp:
        ip = os.environ.get("MASTER_ADDR", "128.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)

    if args.task_name == 'forecast':
        Exp = Exp_Forecast
    else:
        Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)  # set experiments
            if getattr(args, 'use_moe', False) and hasattr(exp, 'model') \
                    and hasattr(exp.model, 'convert_dense_ffn_to_moe'):
                exp.model.convert_dense_ffn_to_moe()
            setting = '{}_{}_{}_{}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.input_token_len,
                args.output_token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.e_layers,
                args.d_model,
                args.d_ff,
                args.n_heads,
                args.cosine,
                args.des, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if not args.ddp and not args.dp:
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.input_token_len,
            args.output_token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.e_layers,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.cosine,
            args.des, ii)
        exp = Exp(args)  # set experiments
        if getattr(args, 'use_moe', False) and hasattr(exp, 'model') \
                and hasattr(exp.model, 'convert_dense_ffn_to_moe'):
            exp.model.convert_dense_ffn_to_moe()
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
