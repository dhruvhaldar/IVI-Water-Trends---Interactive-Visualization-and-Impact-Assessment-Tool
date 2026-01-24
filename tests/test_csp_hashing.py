
import pytest
from ivi_water.security_utils import inject_csp_meta_tag

# Check if BS4 is available
try:
    import bs4
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

def test_inject_csp_fragment():
    """Test CSP injection on HTML fragment (no head/html tags)."""
    fragment = "<div><script>console.log('test')</script></div>"
    result = inject_csp_meta_tag(fragment)

    assert "Content-Security-Policy" in result

    if HAS_BS4:
        # Should use hash-based CSP even for fragments
        assert "sha256-" in result
        assert "script-src 'unsafe-inline'" not in result
    else:
        assert "script-src 'unsafe-inline'" in result

def test_inject_csp_full_html():
    """Test CSP injection on full HTML."""
    html = "<html><head></head><body><script>console.log('test')</script></body></html>"
    result = inject_csp_meta_tag(html)

    assert "Content-Security-Policy" in result

    if HAS_BS4:
        # Should use hash-based CSP
        assert "sha256-" in result
        assert "script-src 'unsafe-inline'" not in result
    else:
        assert "script-src 'unsafe-inline'" in result

def test_inject_csp_no_scripts():
    """Test CSP injection when no scripts are present."""
    html = "<html><body><p>Hello</p></body></html>"
    result = inject_csp_meta_tag(html)

    assert "Content-Security-Policy" in result

    # Even without scripts, we should try to be strict if possible.
    # But current implementation iterates scripts. If no scripts, hashes is empty.
    # Then `if hashes or sources`... sources has 'self'.
    # So it enters the block and creates strict CSP (script-src 'self').

    if HAS_BS4:
        assert "script-src 'self'" in result
        assert "script-src 'unsafe-inline'" not in result
    else:
        assert "script-src 'unsafe-inline'" in result
