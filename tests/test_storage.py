import math

from raglab.storage.postgres import _vector


def test_vector_serialization_is_explicit_and_precise() -> None:
    assert _vector([1.0, -0.25, math.pi]) == "[1,-0.25,3.1415926535897931]"
