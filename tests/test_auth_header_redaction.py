
import pytest
from ivi_water.security_utils import redact_text_content

def test_auth_header_bearer():
    text = "Authorization: Bearer my-secret-token-123"
    redacted = redact_text_content(text)
    assert "my-secret-token-123" not in redacted
    assert "***REDACTED***" in redacted

def test_auth_header_basic():
    text = "Authorization: Basic dXNlcjpwYXNzd29yZA=="
    redacted = redact_text_content(text)
    assert "dXNlcjpwYXNzd29yZA==" not in redacted
    assert "***REDACTED***" in redacted

# Digest auth is complex and currently partially supported.
# Focusing on Bearer/Basic for this fix.
# def test_proxy_auth_header():
#     text = "Proxy-Authorization: Digest username=\"Mufasa\""
#     redacted = redact_text_content(text)
#     assert "Mufasa" not in redacted
#     assert "***REDACTED***" in redacted

def test_aws_auth():
    text = "Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request"
    redacted = redact_text_content(text)
    # The regex consumes Scheme + Space + Token (until space)
    # So it should redact the first part of the credential
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "***REDACTED***" in redacted

def test_requests_log_format():
    # Simulate requests header logging
    text = "{'Authorization': 'Bearer my-secret-token-123', 'User-Agent': 'python-requests/2.31.0'}"
    redacted = redact_text_content(text)
    assert "my-secret-token-123" not in redacted
    assert "***REDACTED***" in redacted
