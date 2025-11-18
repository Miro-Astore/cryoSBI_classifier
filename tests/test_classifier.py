import pytest
import json
import torch
from cryo_sbi.inference.models import build_models
from cryo_sbi.inference.models import estimator_models
from cryo_sbi.inference.check_train_config import check_train_params


@pytest.fixture(
    params=[
        "tests/config_files/training_params_mlp.json",
        "tests/config_files/training_params_proto.json",
    ]
)
def train_params(request):
    config = json.load(open(request.param))
    return check_train_params(config)


def test_build_classifier_model(train_params):
    print(train_params)
    posterior_model = build_models.build_classifier(train_params)
    assert isinstance(posterior_model, estimator_models.ClassifierWithEmbedding)


@pytest.mark.parametrize(
    ("batch_size", "sample_size"), [(1, 1), (2, 10), (5, 1000), (100, 2)]
)
def test_sample_npe_model(train_params, batch_size, sample_size):
    posterior_model = build_models.build_classifier(train_params)
    test_image = torch.randn((batch_size, 128, 128))
    logits = posterior_model(test_image)
    assert logits.shape == torch.Size(
        [batch_size, train_params["CLASSIFIER"]["NUM_CLASSES"]]
    )
