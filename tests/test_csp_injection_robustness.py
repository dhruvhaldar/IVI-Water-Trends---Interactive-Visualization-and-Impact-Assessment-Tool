
import pytest
import re
from ivi_water.security_utils import inject_csp_meta_tag

class TestCSPInjectionRobustness:
    """Test suite for robust CSP injection in various HTML structures."""

    def test_basic_head_injection(self):
        """Test injection into standard <head>."""
        html = "<html><head><title>Test</title></head><body></body></html>"
        result = inject_csp_meta_tag(html)
        assert '<meta http-equiv="Content-Security-Policy"' in result
        assert result.count("<head>") == 1
        # Should be inside head
        assert re.search(r"<head>.*<meta.*Content-Security-Policy.*<title>", result, re.DOTALL)

    def test_uppercase_tags(self):
        """Test injection with uppercase tags."""
        html = "<HTML><HEAD><TITLE>Test</TITLE></HEAD><BODY></BODY></HTML>"
        result = inject_csp_meta_tag(html)
        assert '<meta http-equiv="Content-Security-Policy"' in result
        # Should preserve case of original tags but inject CSP
        assert "<HEAD>" in result
        assert result.count("<HEAD>") == 1  # No duplicate heads
        assert result.count("<head>") == 0  # No new lowercase head mixed in

        # CSP should be after HEAD
        head_pos = result.find("<HEAD>")
        csp_pos = result.find('<meta http-equiv="Content-Security-Policy"')
        assert head_pos < csp_pos

    def test_tags_with_attributes(self):
        """Test injection with attributes in tags."""
        html = '<html lang="en"><head profile="http://example.org"><title>Test</title></head><body></body></html>'
        result = inject_csp_meta_tag(html)
        assert '<meta http-equiv="Content-Security-Policy"' in result
        assert '<head profile="http://example.org">' in result
        assert result.count("<head") == 1

        # CSP should be after head tag
        match = re.search(r'<head[^>]*>\s*<meta http-equiv="Content-Security-Policy"', result)
        assert match is not None

    def test_missing_head_with_html(self):
        """Test injection when <head> is missing but <html> is present."""
        html = "<html><body><h1>Hello</h1></body></html>"
        result = inject_csp_meta_tag(html)
        assert '<meta http-equiv="Content-Security-Policy"' in result
        assert "<head>" in result
        assert "</head>" in result

        # Should act as a wrapper around CSP, inserted after <html>
        match = re.search(r"<html>\s*<head>\s*<meta.*Content-Security-Policy.*</head>", result, re.DOTALL)
        assert match is not None

    def test_missing_head_with_html_attributes(self):
        """Test injection when <head> is missing but <html ...> is present."""
        html = '<html lang="en" dir="ltr"><body><h1>Hello</h1></body></html>'
        result = inject_csp_meta_tag(html)
        assert '<meta http-equiv="Content-Security-Policy"' in result
        assert "<head>" in result

        # Should insert after <html ...>
        match = re.search(r'<html[^>]*>\s*<head>\s*<meta.*Content-Security-Policy', result, re.DOTALL)
        assert match is not None

    def test_no_structure(self):
        """Test injection when no <html> or <head> tags exist (fragment)."""
        html = "<div>Just a fragment</div>"
        result = inject_csp_meta_tag(html)
        assert '<meta http-equiv="Content-Security-Policy"' in result
        # Should prepend
        assert result.startswith('<meta http-equiv="Content-Security-Policy"')

    def test_already_present(self):
        """Test that CSP is not injected if already present."""
        html = '<html><head><meta http-equiv="Content-Security-Policy" content="..."></head></html>'
        result = inject_csp_meta_tag(html)
        assert result.count('<meta http-equiv="Content-Security-Policy"') == 1
