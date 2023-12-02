import pytest
import pandas as pd
from src.env.environment import StockEnvTrade

TEST_DATA_FILE = "data/yahoo_finance_test.csv"

@pytest.fixture(params=[
    pd.DataFrame({"A":[1,2,3],"B":[2,3,5],"C":[100, 200, 300], "D":[300, 200, 100], "E":[1,8, 10], "F":[200, 100, 150],"G":[190, 185, 191], "H":[90, 100, 110]}),
    pd.read_csv(TEST_DATA_FILE),

])
def input_stock_data(request):
    return request.param


@pytest.fixture
def test_envs(input_stock_data):
    test_env_1 = StockEnvTrade(input_stock_data)
    test_env_2 = StockEnvTrade(input_stock_data)
    return [test_env_1, test_env_2]


@pytest.fixture
def test_env(input_stock_data):
    test_env = StockEnvTrade(input_stock_data)
    return test_env


