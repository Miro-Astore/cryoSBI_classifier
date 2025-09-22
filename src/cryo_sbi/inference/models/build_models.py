import torch.nn as nn
from functools import partial
import zuko
import lampe
import cryo_sbi.inference.models.estimator_models as estimator_models
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


def build_classifier(config: dict, **embedding_kwargs) -> nn.Module:
    """
    Function to build NPE estimator with embedding net
    from config_file

    Args:
        config (dict): config file
        embedding_kwargs (dict): kwargs for embedding net

    Returns:
        estimator (nn.Module): NPE estimator
    """

    try:
        embedding = partial(
            EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"], **embedding_kwargs
        )
    except KeyError:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )
    
    try:
        prototype = config["PROTOTYPE"]
    except KeyError:
        prototype = False

    estimator = estimator_models.ClassifierWithEmbedding(
        embedding_net=embedding,
        output_embedding_dim=config["OUT_DIM"],
        num_classes=config["NUM_CLASSES"],
        num_layers=config["NUM_LAYERS"],
        nodes_per_layer=config["NODES_PER_LAYER"],
        prototype=prototype,
        **{
            "activation": partial(nn.LeakyReLU, 0.1),
            "dropout": config["DROPOUT"],
        },
    )

    return estimator
