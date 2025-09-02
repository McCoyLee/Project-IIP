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
# 说明：将 ETT 系列映射到通用多变量基准数据集，便于直接读取 ETTm1/ETTm2/ETTh1/ETTh2 的 CSV
data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy,

    # === 新增：ETT 系列 ===
    'ETTm1': MultivariateDatasetBenchmark,
    'ETTm2': MultivariateDatasetBenchmark,
    'ETTh1': MultivariateDatasetBenchmark,
    'ETTh2': MultivariateDatasetBenchmark,
    'Weather': MultivariateDatasetBenchmark,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise KeyError(f"Unknown dataset key '{args.data}'. "
                       f"Available: {list(data_dict.keys())}")
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

    # 构造数据集（train/val 用训练窗口；test/pred 用测试窗口）
    if flag in ['train', 'val']:
        size = [args.seq_len, args.input_token_len, args.output_token_len]
    else:  # 'test' or 'pred'
        size = [args.test_seq_len, args.input_token_len, args.test_pred_len]

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        nonautoregressive=args.nonautoregressive,
        test_flag=args.test_flag,
        subset_rand_ratio=args.subset_rand_ratio,
    )

    print(flag, len(data_set))

    # 仅在训练阶段且 num_workers>0 时启用 persistent_workers（避免 eval 时 num_workers=0 报错/卡住）
    use_persistent = (flag == 'train') and (args.num_workers > 0)

    if args.ddp:
        sampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            persistent_workers=use_persistent,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=use_persistent,
            pin_memory=True,
            drop_last=drop_last,
        )
    return data_set, data_loader

