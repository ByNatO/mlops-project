import pytest
import pandas as pd
from src.data.validate import validate_missing, validate_data_quality

def test_validate_missing():
    df = pd.DataFrame({'A': [1,2,None], 'B': [4,5,6]})
    with pytest.raises(ValueError):
        validate_missing(df, threshold=0.05)
    df2 = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    assert validate_missing(df2, threshold=0.05) is True

def test_validate_data_quality():
    df = pd.read_csv('data/raw/creditcard.csv')
    assert validate_data_quality(df) is True