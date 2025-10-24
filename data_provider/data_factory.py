# data_provider/data_factory.py
from data_provider.data_loader import (
    UnivariateDatasetBenchmark,
    MultivariateDatasetBenchmark,
    Global_Temp,
    Global_Wind,
    Dataset_ERA5_Pretrain,
    Dataset_ERA5_Pretrain_Test,
    UTSD,
    UTSD_Npy,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# 可用数据集注册表
# 说明：
# - 将 ETT 系列 + Weather + Electricity + Exchange + Solar
#   都映射到通用多变量基准数据集 MultivariateDatasetBenchmark，
#   便于直接读取“首列时间/后续为数值列”的 CSV。
data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy,

    # === ETT 系列（CSV：date + 变量列） ===
    'ETTm1': MultivariateDatasetBenchmark,
    'ETTm2': MultivariateDatasetBenchmark,
    'ETTh1': MultivariateDatasetBenchmark,
    'ETTh2': MultivariateDatasetBenchmark,

    # === 其它多变量基准（CSV：date + 变量列）===
    'Weather': MultivariateDatasetBenchmark,
    'Electricity': MultivariateDatasetBenchmark,
    'Exchange': MultivariateDatasetBenchmark,
    'Solar': MultivariateDatasetBenchmark,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise KeyError(
            f"Unknown dataset key '{args.data}'. "
            f"Available: {list(data_dict.keys())}"
        )

    Data = data_dict[args.data]

    # Loader 行为
    if flag in ['test', 'val', 'pred']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:  # 'train'
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    # 构造滑窗长度（train/val 与 test/pred 可不同）
    if flag in ['train', 'val']:
        size = [args.seq_len, args.input_token_len, args.output_token_len]
    else:  # 'test' or 'pred'
        # 若工程里没有 test_seq_len / test_pred_len，也可与训练保持一致
        test_seq_len = getattr(args, 'test_seq_len', args.seq_len)
        test_pred_len = getattr(args, 'test_pred_len', args.output_token_len)
        size = [test_seq_len, args.input_token_len, test_pred_len]

    # 实例化数据集
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        nonautoregressive=getattr(args, 'nonautoregressive', False),
        test_flag=getattr(args, 'test_flag', False),
        subset_rand_ratio=getattr(args, 'subset_rand_ratio', 1.0),
    )

    print(flag, len(data_set))

    # 仅在训练阶段且 num_workers>0 时启用 persistent_workers
    use_persistent = (flag == 'train') and (getattr(args, 'num_workers', 0) > 0)

    if getattr(args, 'ddp', False):
        sampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=getattr(args, 'num_workers', 0),
            persistent_workers=use_persistent,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=getattr(args, 'num_workers', 0),
            persistent_workers=use_persistent,
            pin_memory=True,
            drop_last=drop_last,
        )

    return data_set, data_loader
