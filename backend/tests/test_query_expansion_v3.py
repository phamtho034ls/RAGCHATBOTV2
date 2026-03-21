import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.query_expansion import should_expand_query_v2


def test_expand_keyword_signal():
    assert should_expand_query_v2("Liệt kê các điều cấm", []) is True


def test_expand_false_empty():
    assert should_expand_query_v2("a", []) is False
