## 2024-05-24 - [Initial Security Scan]
**Vulnerability:** XSS in HTML Report Generation
**Learning:** Dynamic content in generated reports, even column names, must always be escaped.
**Prevention:** Use `html.escape()` for all dynamic strings in HTML templates.
