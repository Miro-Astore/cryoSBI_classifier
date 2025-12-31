from copy import deepcopy

default_train_config = {
    "EMBEDDING": {
        "MODEL": "RESNET18",
        "OUT_DIM": 128,
    },
    "CLASSIFIER": {
        "MODEL": "MLP",
        "NUM_CLASSES": 44,
        "NUM_LAYERS": 3,
        "NODES_PER_LAYER": 128,
        "DROPOUT": 0.1,
    },
    "LEARNING_RATE": 0.0005,
    "ONE_CYCLE_SCHEDULER": True,
    "CLIP_GRADIENT": 5.0,
    "WEIGHT_DECAY": 0.001,
    "BATCH_SIZE": 128,
}


def check_train_params(config: dict) -> dict:
    merged = deepcopy(default_train_config)

    for k, v in config.items():
        if k == "CLASSIFIER" and isinstance(v, dict):
            has_model = "MODEL" in v
            has_num_classes = "NUM_CLASSES" in v
            if not (has_model and has_num_classes):
                merged[k].update(v)
            else:
                merged[k] = v
        elif k == "EMBEDDING" and isinstance(v, dict):
            merged[k].update(v)
        else:
            merged[k] = v

    emb = merged.get("EMBEDDING", {})
    assert "MODEL" in emb, "EMBEDDING.MODEL required"
    assert "OUT_DIM" in emb, "EMBEDDING.OUT_DIM required"

    clf = merged.get("CLASSIFIER", {})
    assert "MODEL" in clf, "CLASSIFIER.MODEL required"
    assert "NUM_CLASSES" in clf, "CLASSIFIER.NUM_CLASSES required"

    return merged

def check_image_params(config: dict) -> dict:
    needed_keys = [
        "N_PIXELS",
        "PIXEL_SIZE",
        "SIGMA",
        "SHIFT",
        "DEFOCUS",
        "SNR",
        "MODEL_FILE",
        "AMP",
        "B_FACTOR",
    ]

    for key in needed_keys:
        assert key in config.keys(), f"Please provide a value for {key}"

    return config