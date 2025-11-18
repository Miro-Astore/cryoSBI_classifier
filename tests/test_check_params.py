import json
from cryo_sbi.wpa_simulator.check_image_config import check_image_params
from cryo_sbi.inference.check_train_config import check_train_params


def test_check_train_params_mlp():
    config = json.load(open("tests/config_files/training_params_mlp.json"))
    merged = check_train_params(config)
    assert merged["EMBEDDING"]["MODEL"] == "RESNET18"
    assert merged["EMBEDDING"]["OUT_DIM"] == 128
    assert merged["CLASSIFIER"]["MODEL"] == "MLP"
    assert merged["CLASSIFIER"]["NUM_CLASSES"] == 44
    assert merged["CLASSIFIER"]["NUM_LAYERS"] == 8
    assert merged["CLASSIFIER"]["NODES_PER_LAYER"] == 128
    assert merged["CLASSIFIER"]["DROPOUT"] == 0.05
    assert merged["LEARNING_RATE"] == 0.0005
    assert merged["ONE_CYCLE_SCHEDULER"] is True
    assert merged["CLIP_GRADIENT"] == 5.0
    assert merged["WEIGHT_DECAY"] == 100
    assert merged["BATCH_SIZE"] == 128


def test_check_train_params_proto():
    config = json.load(open("tests/config_files/training_params_proto.json"))
    merged = check_train_params(config)
    assert merged["EMBEDDING"]["MODEL"] == "RESNET18"
    assert merged["EMBEDDING"]["OUT_DIM"] == 128
    assert merged["CLASSIFIER"]["MODEL"] == "PROTOTYPE"
    assert merged["CLASSIFIER"]["NUM_CLASSES"] == 44
    assert merged["LEARNING_RATE"] == 0.0005
    assert merged["ONE_CYCLE_SCHEDULER"] is True
    assert merged["CLIP_GRADIENT"] == 5.0
    assert merged["WEIGHT_DECAY"] == 0.01
    assert merged["BATCH_SIZE"] == 32


def test_check_image_params():
    config = json.load(open("tests/config_files/image_params_testing.json"))
    config = check_image_params(config)
