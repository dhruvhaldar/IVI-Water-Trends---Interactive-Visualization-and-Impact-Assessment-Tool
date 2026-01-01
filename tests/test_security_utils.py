
import pytest
import json
from ivi_water.security_utils import redact_sensitive_data, hash_data

def test_redact_sensitive_data_dict():
    data = {
        'username': 'user1',
        'password': 'secret_password',
        'api_key': 'abc12345',
        'nested': {
            'token': 'secret_token',
            'public': 'data'
        }
    }

    redacted = redact_sensitive_data(data)

    assert redacted['username'] == 'user1'
    assert redacted['password'] == '[REDACTED]'
    assert redacted['api_key'] == '[REDACTED]'
    assert redacted['nested']['token'] == '[REDACTED]'
    assert redacted['nested']['public'] == 'data'

def test_redact_sensitive_data_list():
    data = [
        {'id': 1, 'secret': 'hidden'},
        {'id': 2, 'auth': 'credentials'}
    ]

    redacted = redact_sensitive_data(data)

    assert redacted[0]['id'] == 1
    assert redacted[0]['secret'] == '[REDACTED]'
    assert redacted[1]['id'] == 2
    assert redacted[1]['auth'] == '[REDACTED]'

def test_redact_sensitive_data_case_insensitive():
    data = {'API_KEY': '123', 'PaSsWoRd': 'abc'}
    redacted = redact_sensitive_data(data)
    assert redacted['API_KEY'] == '[REDACTED]'
    assert redacted['PaSsWoRd'] == '[REDACTED]'

def test_redact_does_not_mutate_original():
    data = {'password': 'secret'}
    redacted = redact_sensitive_data(data)
    assert data['password'] == 'secret'
    assert redacted['password'] == '[REDACTED]'

def test_hash_data_consistency():
    data1 = {'a': 1, 'b': 2}
    data2 = {'b': 2, 'a': 1}  # Different order

    # Hashes should be identical despite key order
    assert hash_data(data1) == hash_data(data2)

def test_hash_data_uniqueness():
    data1 = {'a': 1}
    data2 = {'a': 2}

    assert hash_data(data1) != hash_data(data2)

def test_hash_data_types():
    assert hash_data("string")
    assert hash_data(123)
    assert hash_data([1, 2, 3])
