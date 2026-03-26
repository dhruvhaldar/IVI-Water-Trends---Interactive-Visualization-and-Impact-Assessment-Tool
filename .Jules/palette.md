## 2023-10-25 - HTML Report Accessibility
**Learning:** Generated HTML reports lacked viewport meta tags and semantic structure, making them poorly responsive on mobile and harder for screen readers to navigate.
**Action:** Added `<meta name="viewport">` for mobile responsiveness and semantic HTML tags (`<main>`, `<header>`, `<section>`) with `aria-labelledby` to improve screen reader navigation.
## 2025-03-26 - Interactive Plotly HTML Mobile Responsiveness
**Learning:** Plotly's `to_html()` default exports lack viewport meta tags, causing interactive charts to appear zoomed out or require pinching on mobile devices, diminishing the responsive experience.
**Action:** Injected `<meta name="viewport" content="width=device-width, initial-scale=1.0">` alongside title injection when generating HTML exports from Plotly figures to ensure correct mobile scaling.
