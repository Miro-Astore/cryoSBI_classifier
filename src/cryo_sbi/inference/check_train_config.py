from copy import deepcopy

default_train_config = {
    "EMBEDDING": {
        "MODEL": "RESNET18",
        "OUT_DIM": 128,
    },
    "CLASSIFIER": {
        "MODEL": "MLP",
        "NUM_CLASSES": 44,
        "NUM_LAYERS": 8,
        "NODES_PER_LAYER": 128,
        "DROPOUT": 0.05,
    },
    "LEARNING_RATE": 0.0005,
    "ONE_CYCLE_SCHEDULER": True,
    "CLIP_GRADIENT": 5.0,
    "WEIGHT_DECAY": 0.001,
    "BATCH_SIZE": 128,
}


def check_train_params(config: dict) -> dict:
    merged = deepcopy(default_train_config)

    # Merge top-level keys
    for k, v in config.items():
        if k == "CLASSIFIER" and isinstance(v, dict):
            # Only merge default if MODEL or NUM_CLASSES missing
            has_model = "MODEL" in v
            has_num_classes = "NUM_CLASSES" in v
            if not (has_model and has_num_classes):
                merged[k].update(v)
            else:
                merged[k] = v  # leave user CLASSIFIER as is
        elif k == "EMBEDDING" and isinstance(v, dict):
            merged[k].update(v)
        else:
            merged[k] = v

    # Ensure required keys
    emb = merged.get("EMBEDDING", {})
    assert "MODEL" in emb, "EMBEDDING.MODEL required"
    assert "OUT_DIM" in emb, "EMBEDDING.OUT_DIM required"

    clf = merged.get("CLASSIFIER", {})
    assert "MODEL" in clf, "CLASSIFIER.MODEL required"
    assert "NUM_CLASSES" in clf, "CLASSIFIER.NUM_CLASSES required"

    return merged
