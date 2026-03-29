from avalanche.models import SimpleMLP
from models.base_multi_head_model import BaseMultiHeadModel


def build_model(model_name: str, **kwargs):
    model_name = model_name.lower()
    
    if model_name == "avalanche_simplemlp":
        return SimpleMLP(**kwargs)
    
    if model_name == "base_multi_head":
        return BaseMultiHeadModel(**kwargs)

    raise ValueError(f"Unknown model_name: {model_name}")