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
## 2024-10-18 - HTML Table Responsiveness and Interactive States
**Learning:** Pandas DataFrame HTML tables generated directly in reports often break layouts on narrow mobile screens and lack clear visual distinction for reading and interactive elements.
**Action:** Always wrap data tables in a `.table-responsive` container with `overflow-x: auto`. Additionally, ensure tables have zebra striping (`tr:nth-child(even)`) and hover states (`tr:hover`) for improved readability, and that hidden skip-to-content links utilize `:focus-visible` with high-contrast outlines (e.g., `#ff7f0e`) for robust keyboard navigation accessibility.
## 2024-10-18 - Scrollable Container Keyboard Accessibility
**Learning:** HTML elements with `overflow` (like `.table-responsive`) trap keyboard-only users who cannot scroll them horizontally because the container itself cannot receive focus.
**Action:** Always add `tabindex="0"`, `role="region"`, and an `aria-labelledby` or `aria-label` attribute to scrollable containers, along with a `:focus-visible` outline for clear visual feedback.
## 2024-11-20 - Interactive Plotly Chart Accessibility
**Learning:** Plotly interactive charts exported to HTML generate a container div (`<div class="plotly-graph-div">`) that lacks screen reader and keyboard accessibility attributes, leaving the visualization opaque and inaccessible to non-visual users navigating the document.
**Action:** Always inject `role="region"`, a descriptive `aria-label`, and `tabindex="0"` into the `plotly-graph-div` container when generating standalone HTML or reports with Plotly. This allows screen readers to announce the interactive area and keyboard users to focus on it.

## 2024-04-03 - Focus Visible for Third-Party Components
**Learning:** Injecting `tabindex="0"` into third-party component wrappers (like Plotly's HTML export) is not enough for keyboard accessibility. You must explicitly inject corresponding `:focus-visible` CSS as well, because the library's default styles will not cover custom accessibility enhancements. Without it, the focus indicator will not appear.
**Action:** Always pair `tabindex="0"` additions with `:focus-visible` CSS rules when enhancing third-party component wrappers.
