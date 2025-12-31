
import unittest
from ivi_water.security_utils import redact_sensitive_data, hash_data

class TestSecurityUtils(unittest.TestCase):

    def test_redact_sensitive_data_dict(self):
        data = {
            'username': 'jdoe',
            'api_key': 'secret123',
            'password': 'password123',
            'config': {
                'token': 'token456',
                'public': 'visible'
            }
        }

        redacted = redact_sensitive_data(data)

        self.assertEqual(redacted['username'], 'jdoe')
        self.assertEqual(redacted['api_key'], '***REDACTED***')
        self.assertEqual(redacted['password'], '***REDACTED***')
        self.assertEqual(redacted['config']['token'], '***REDACTED***')
        self.assertEqual(redacted['config']['public'], 'visible')

    def test_redact_sensitive_data_list(self):
        data = [
            {'api_key': 'secret1'},
            {'public': 'data'}
        ]

        redacted = redact_sensitive_data(data)

        self.assertEqual(redacted[0]['api_key'], '***REDACTED***')
        self.assertEqual(redacted[1]['public'], 'data')

    def test_hash_data(self):
        data1 = {'a': 1, 'b': 2}
        data2 = {'b': 2, 'a': 1} # Same data, different order

        hash1 = hash_data(data1)
        hash2 = hash_data(data2)

        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64) # SHA-256 length

if __name__ == '__main__':
    unittest.main()
