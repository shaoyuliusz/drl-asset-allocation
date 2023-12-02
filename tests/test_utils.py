import pytest
from collections import namedtuple
from src.utils.common_utils import replace_dict, dict_to_namedtuple

@pytest.fixture
def dict_data():
    old_dict = {"a": {"a1":1, "a2":2}, "b": {"b1": 3}}
    new_dict = {"a1":100, "a2": 200, "b1":300}
    outcome_dict = {"a": {"a1":100, "a2":200}, "b": {"b1": 300}}
    return old_dict, new_dict, outcome_dict

@pytest.fixture
def namedtuple_data():
    true_dict = {"a": {"a1":1, "a2":2}, "b": {"b1": 3}}
    dt = namedtuple("Params", ["a", "b"])
    sub1 = namedtuple("Params_A", ["a1", "a2"])
    sub2 = namedtuple("Params_B", ["b1"])
    sub1_obj = sub1(1, 2)
    sub2_obj = sub2(3)
    namedtuple_dict = dt(sub1_obj, sub2_obj)
    return true_dict, namedtuple_dict

def test_replace_dict(dict_data):
    old_dict, new_dict, outcome_dict = dict_data
    pred_dict = replace_dict(old_dict, new_dict)

    assert pred_dict == outcome_dict

def test_dict_to_namedtuple(namedtuple_data):
    true_dict, namedtuple_dict = namedtuple_data

    assert dict_to_namedtuple(true_dict) == namedtuple_dict