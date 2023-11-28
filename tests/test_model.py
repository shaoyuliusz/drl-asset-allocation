import pytest
import pandas as pd
@pytest.fixture
def dummy_stock_data():
    stocks = pd.DataFrame({'AAPL':[50, 100, 150], "JPM": [100, 50, 20]})
    return stocks

def test_drl_model():
    pass