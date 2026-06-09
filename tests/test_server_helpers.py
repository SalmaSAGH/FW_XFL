import pickle
from server.server import weights_to_base64, base64_to_weights
from collections import OrderedDict


def test_weights_serialization_roundtrip():
    weights = OrderedDict({"layer1": 1, "layer2": [1, 2, 3]})
    encoded = weights_to_base64(weights)
    decoded = base64_to_weights(encoded)
    assert decoded == weights
