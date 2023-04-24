"""
File that converts the name to function
"""
from typing import Dict

from torch import nn, optim

NAME_TO_LOSS: Dict = {
    "NLLLoss": nn.NLLLoss,
    "MAE": nn.L1Loss,
    "MSE": nn.MSELoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "HingeEmbedding": nn.HingeEmbeddingLoss,
    "MarginRanking": nn.MarginRankingLoss,
    "TripletMargin": nn.TripletMarginLoss,
    "KLDiv": nn.KLDivLoss,
}


NAME_TO_OPTIMIZER: Dict = {
    "Adam": optim.AdamW,
    "SGD": optim.SGD,
}
