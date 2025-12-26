## 2025-12-26 - Path Traversal Prevention in File Exports
**Vulnerability:** User-controlled filenames in CLI commands and `DataProcessor.export_processed_data` were not properly sanitized, allowing path traversal (e.g., `../evil_file.csv`).
**Learning:** Even when `click` validates input types, it does not sanitize paths. Standard library `os.path.basename` is a good start, but a strict whitelist is safer for filenames.
**Prevention:** Always use `sanitize_filename` (which whitelists alphanumeric, `_`, `-`, `.`) for any user-provided filename before using it in file operations.
