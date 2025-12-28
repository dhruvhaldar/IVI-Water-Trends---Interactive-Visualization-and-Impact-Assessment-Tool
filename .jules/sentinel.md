## 2024-05-24 - [XSS in HTML Reports]
**Vulnerability:** Detected XSS vulnerability in HTML report generation where DataFrame column names were injected directly into HTML without escaping.
**Learning:** Pandas `to_html` handles cell content escaping but manual string interpolation for report metadata (like column lists) is vulnerable.
**Prevention:** Always use `html.escape()` when manually constructing HTML strings, even for seemingly safe data like column headers.
