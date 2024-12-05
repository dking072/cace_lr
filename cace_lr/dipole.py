import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Tuple, Union
from cace.tools import scatter_sum

class Dipoles(nn.Module):
    def __init__(self,feature_key : str ='q'):
        super().__init__()
        self.feature_key = feature_key
        self.model_outputs = ["pred_dipole"]

    def forward(self, data: Dict[str, torch.Tensor], 
                training: bool = False, output_index: int = None) -> Dict[str, torch.Tensor]:
        dipole = data["positions"] * data[self.feature_key]
        dipole = scatter_sum(src=dipole, index=data["batch"], dim=0)
        data["pred_dipole"] = dipole
        return data 