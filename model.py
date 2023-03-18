import torch
from torch.nn import Linear
from transformers.utils.dummy_pt_objects import BertModel

class CLS_Model(torch.nn.Module):
    """
    BERTを使用したテキスト分類モデル

    Attributes:
    -----------
    checkpoint : str
        使用するBERTのチェックポイント名
    bert : transformers.BertModel
        BERT
    hidden_dim : int
        BERTの特徴量の次元
    output_dim : int
        出力の次元
    output_layer : nn.Linear
        hidden_dim -> output_dim に変換する全結合層
    """
    def __init__(self, output_dim:int, bert_checkpoint='bert-base-uncased') -> None:
        super().__init__()
        self.checkpoint = bert_checkpoint
        self.bert:BertModel = BertModel.from_pretrained(bert_checkpoint)
        self.hidden_dim:int = self.bert.config.hidden_size
        self.output_dim:int = output_dim
        self.output_layer = Linear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            bias=True
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        順伝播の計算

        Parameters:
        -----------
        X : dict[str, torch.Tensor]
            transformersのtokenizerの出力と同じ
            dictのkeyは['input_ids', 'attention_mask', 'token_type_ids']の3つ
        """
        bert_output = self.bert(**X).last_hidden_state[:, 0, :] # [CLS]の最終層の出力だけ使う
        logits = self.output_layer(bert_output)

        return logits

# 動作するかテスト
if __name__ == '__main__':
    model = CLS_Model(output_dim=2)
    X = {'input_ids':
            torch.zeros(2, 3).long(),
         'attention_mask':
            torch.ones(2, 3),
         'token_type_ids':
            torch.zeros(2, 3).long()}
    print(model(X).shape)
