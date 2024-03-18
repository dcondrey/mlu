import numpy as np
from modules.array_manipulation.array_manipulation import filter, aggregate, summary

def test_filter():
    array = np.array([1, 2, 3, 4])
    result = filter(array, lambda x: x > 2)
    np.testing.assert_array_equal(result, np.array([3, 4]))

def test_aggregate():
    array = np.array([1, 2, 3, 4])
    assert aggregate(array, 'sum') == 10
    assert aggregate(array, 'mean') == 2.5
    assert aggregate(array, 'max') == 4
    assert aggregate(array, 'min') == 1

def test_summary():
    array = np.array([1, 2, 3, 4])
    result = summary(array)
    assert result['count'] == 4
    assert result['mean'] == 2.5
    assert result['std'] != 0
    assert result['min'] == 1
    assert result['max'] == 4
    assert '25%' in result and '50%' in result and '75%' in result