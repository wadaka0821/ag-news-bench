import random
import argparse
import time

import numpy as np
from numpy.typing import NDArray
import torch
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer 
from datasets import load_dataset
from tqdm import tqdm

from model import CLS_Model
from utils import (
    DEVICE, 
    ExperimentConfig, 
    Result,
    InvalidCommandLineArgument,
    save_config,
    save_result
)

def fix_seed(seed:int):
    """
    再現性のためにシード値を固定
    python, numpy, pytorch のシード値を固定

    Parameters
    ----------
    seed : int
        指定したいシード値
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def get_time() -> float:
    """
    現在の時刻を取得
    """
    torch.cuda.synchronize() # GPU を使用したときに必要
    return time.time()

def train(config:ExperimentConfig) -> Result:
    """
    モデルの学習と時間計測

    Parameters
    ----------
    config : ExperimentConfig
        実験設定を保持するオブジェクト

    Returns
    -------
    res : Result
        実験結果(各処理にかかった時間)を保持するオブジェクト
    """
    times:list[float] = list() # 各ポイントの時刻を保存

    times.append(get_time()) # トークナイザをロードする時間を計測

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    times.append(get_time()) # モデルをロード(GPU 使用時には GPU メモリへの移動も込み)する時間を計測

    model = CLS_Model(output_dim=4).to(config.device)

    times.append(get_time()) # データのロードからデータローダーの作成までの時間を計測

    # AG-News の学習データの内，10000件を使用
    dataset = load_dataset(
        "ag_news", 
        split = 'train[:10000]'
    )
    # エンコード
    dataset = dataset.map(
        lambda x: tokenizer(
            x['text'], 
            max_length=512, 
            padding=True, 
            truncation=True
        ), 
        batched=True,
        load_from_cache_file=False, # type: ignore
        batch_size = config.batch_size
    )
    # DataLoader への入力用に変換
    dataset.set_format(
        type='torch', 
        columns=['input_ids', 
                 'token_type_ids', 
                 'attention_mask', 
                 'label']
    )

    # DataLoader の作成(num_workersは取り敢えず0)
    if isinstance(dataset, Dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
    else:
        raise Exception(f'type(dataset) need to be Dataset, but {type(dataset)}')

    times.append(get_time()) # 学習にかかる時間を計測

    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()

    for _ in range(config.num_epoch):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            Y = batch['label'].to(config.device)
            X = {key:value.to(config.device) for key, value in batch.items() if key != 'label'}

            logits = model(X)
            loss = criterion(logits, Y)
            loss.backward()

            optimizer.step()
    times.append(get_time())
    times_ndarr:NDArray[np.float32] = np.array(times)
    # times は各ポイントの時刻を表すので times[i] - times[i-1] で処理にかかった時間を計算
    times_diff = times_ndarr[1:] - times_ndarr[:-1]
    res = Result(
        *times_diff.tolist()
    )

    return res

def main(config:ExperimentConfig, base_dir:str) -> None:
    """
    実験のメインの部分

    Parameters
    ---------
    config : ExperimentConfig
        実験設定を保持しているオブジェクト
    base_dir : str
        実験設定，結果を保存するディレクトリ
    """
    res = train(config) # 実験
    save_result(res, config, base_dir) # 結果を保存

def load_arguments() -> tuple[ExperimentConfig, str]:
    """
    コマンドラインからオプションを取得

    Returns
    -------
    config : ExperimentConfig
        実験設定を保持しておくオブジェクト
    base_dir : str
        実験設定，結果を保存するディレクトリ
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_container', type=int, help='起動しているdockerコンテナの数(required)')
    parser.add_argument('--cuda', action='store_true', help='GPU使用の有無(optional)')
    parser.add_argument('--batch_size', type=int, help='バッチサイズ(required)')
    parser.add_argument('--seed', type=int, help='random, numpy, torchのシード値(required)')
    parser.add_argument('--num_epoch', type=int, help='学習するエポック数(optional)')
    parser.add_argument('--base_dir', type=str, help='結果を保存するディレクトリ(required)')

    args = parser.parse_args()
    if not args.num_container:
        raise InvalidCommandLineArgument('num_container')
    if args.cuda and not torch.cuda.is_available(): # --cuda オプションが指定　かつ　GPUが利用可能かチェック
        raise InvalidCommandLineArgument('cuda')
    if not args.batch_size:
        raise InvalidCommandLineArgument('batch_size')
    if not args.seed:
        raise InvalidCommandLineArgument('seed')
    if not args.num_epoch:
        args.num_epoch = 3 # 学習のエポック数はデフォルトで3
    if not args.base_dir:
        args.base_dir = './res'

    device = 'cuda' if args.cuda else 'cpu'
    
    return ExperimentConfig(
        args.num_container,
        device,
        args.batch_size,
        args.seed,
        args.num_epoch
    ), args.base_dir

"""
ディレクトリ構成
experiment
|---res
|    |---cpu
|    |    |---cpu_{num_container1}_{batch_size1}_{seed1}
|    |    |                                       |---exp.conf
|    |    |                                       |---exp.res
|    |    |---cpu_{num_container2}_{batch_size2}_{seed2}
|    |    |                                       |---exp.conf
|    |    |                                       |---exp.res
|    |    | ...
|    |  
|    |---cuda
|         |---cuda_{num_container1}_{batch_size1}_{seed1}
|         |                                       |---exp.conf
|         |                                       |---exp.res
|         |---cuda_{num_container2}_{batch_size2}_{seed2}
|         |                                       |---exp.conf
|         |                                       |---exp.res
|         | ...
|---run.sh
|---exp.py
|---model.py
|---utils.py
"""
if __name__ == '__main__':
    exp_config, base_dir = load_arguments()
    save_config(exp_config, base_dir)
    main(exp_config, base_dir)