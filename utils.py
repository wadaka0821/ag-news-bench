from typing import Literal
import dataclasses
import os
import pickle

DEVICE = Literal['cpu', 'cuda']

# 実験の設定を保持するデータクラス
@dataclasses.dataclass
class ExperimentConfig:
    num_container : int
    device : DEVICE
    batch_size : int
    seed : int
    num_epoch: int = 3

# 実験結果(実行にかかった時間)を保持するデータクラス
@dataclasses.dataclass
class Result:
    initialize_tokenizer : float # トークナイザのロードにかかる時間
    initialize_model : float # モデルのロードにかかる時間
    load_data : float # 学習データのロードにかかる時間
    training : float # モデルの学習にかかる時間

# コマンドラインのオプションの値がおかしいときに使用する例外クラス
class InvalidCommandLineArgument(Exception):
    def __init__(self, arg='') -> None:
        self.arg = arg
    
    def __str__(self) -> str:
        return (
            f"オプション[{self.arg}]が指定されていないか不正な値です\n詳しくは python exp.py -h を実行してください"
        )

def save_config(config:ExperimentConfig, base_dir:str) -> None:
    """
    実験設定を保存

    Parameters
    ----------
    config : ExperimentConfig
        実験設定を保持するオブジェクト
    base_dir : str
        実験設定，結果を保存するディレクトリ
    """
    if not os.path.exists(os.path.join(base_dir, config.device)):
        os.mkdir(os.path.join(base_dir, config.device))
    dir_name = f"{config.device}_{config.num_container}_{config.batch_size}_{config.seed}"
    os.mkdir(os.path.join(base_dir, config.device, dir_name))
    with open(os.path.join(base_dir, config.device, dir_name, 'exp.conf'), 'bw') as f:
        pickle.dump(config, f)

def save_result(res:Result, config:ExperimentConfig, base_dir:str) -> None:
    """
    実験結果を pickle で保存

    Parameters
    ----------
    res : Result
        保存する Result オブジェクト
    config : ExperimentConfig
        実験設定のオブジェクト(保存先のディレクトリの指定に使用)
    base_dir : str
        実験設定，結果を保存するディレクトリ
    """
    dir_name = f"{config.device}_{config.num_container}_{config.batch_size}_{config.seed}"
    with open(os.path.join(base_dir, config.device, dir_name, 'exp.res'), 'bw') as f:
        pickle.dump(res, f)

