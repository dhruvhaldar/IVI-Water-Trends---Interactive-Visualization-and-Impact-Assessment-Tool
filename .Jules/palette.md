## 2023-10-25 - HTML Report Accessibility
**Learning:** Generated HTML reports lacked viewport meta tags and semantic structure, making them poorly responsive on mobile and harder for screen readers to navigate.
**Action:** Added `<meta name="viewport">` for mobile responsiveness and semantic HTML tags (`<main>`, `<header>`, `<section>`) with `aria-labelledby` to improve screen reader navigation.
## 2024-05-15 - HTML Report Skip-to-Content Link
**Learning:** Generated HTML reports with large header or banner sections force screen reader users to listen to repetitive navigation/header content on every view, leading to a frustrating experience.
**Action:** Added a visually hidden "Skip to main content" link at the very top of the `<body>` that becomes visible on `:focus`. This allows keyboard and screen reader users to bypass the header and immediately access the main report data (`#summary-stats-title`).
## 2026-03-29 - Color Contrast on Interactive Elements
**Learning:** The skip-to-content link and section headers used a light blue (`#3498db`) that failed WCAG 2.1 AA color contrast requirements (3.15:1) against white, making it hard to read for users with visual impairments.
**Action:** Darkened the background color to `#226699` to ensure a 6.1:1 contrast ratio, ensuring accessibility for all interactive elements and visual dividers.
## 2024-06-18 - Color-blind Accessibility for Data Visualization
**Learning:** The use of standard red (`#ff6b6b`) and green (`#51cf66`) for comparative charts (With/Without Intervention) creates issues for red-green colorblind users (Deuteranomaly/Protanomaly) and fails WCAG AA contrast ratios against white backgrounds.
**Action:** Replaced the red/green paired colors with a higher-contrast, color-blind friendly scheme using red (`#d62728`) and blue (`#1f77b4`), applying it via `color_discrete_map` to consistently handle accessibility across charts.
