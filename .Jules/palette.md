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

## 2026-04-07 - Dynamic ARIA Labels for Plotly Charts
**Learning:** Hardcoding generic ARIA labels like `aria-label="Interactive Chart"` across all visualizations creates a confusing experience for screen reader users when multiple charts exist on the same page. Without specific context, users cannot distinguish between different data representations.
**Action:** Always extract the dynamic, specific title (e.g., `fig.layout.title.text`), strip any embedded HTML tags, HTML-escape it for safety, and use it in the `aria-label` attribute when injecting accessibility metadata into third-party wrapper elements.

## 2024-10-24 - Number Formatting for Readability
**Learning:** Raw integers displayed in summary tables and data reports (e.g., `1234567`) are difficult for users to parse at a glance and increase cognitive load compared to properly formatted numbers.
**Action:** Always apply thousands separators (e.g., via f-string formatting like `f"{value:,}"`) to large numerical values in data tables, summaries, and HTML text output to improve quick scanning and readability.
## 2024-05-24 - Consistent Semantic Color Mapping Across Dashboards
**Learning:** When generating multiple visualizations that share a categorical dimension (e.g., seasons), omitting explicit color mapping (like `color_discrete_map`) in some charts causes plotting libraries to assign default palette colors. This inconsistency creates significant cognitive friction for users, as the same color means different things in different contexts.
**Action:** Always apply explicit semantic color mappings (`color_discrete_map`) and category ordering (`category_orders`) consistently across all charts within an application or report.
## 2024-11-20 - Plotly Tooltip Formatting for Readability
**Learning:** Default tooltips in Plotly display raw floating-point numbers or large integers without thousands separators, making it difficult for users to read and quickly parse values in interactive dashboards, creating inconsistency with formatted static reports.
**Action:** When customizing Plotly chart tooltips (`hovertemplate`), always use d3-style formatting syntax (e.g., `%{y:,.2f}` or `%{y:,}`) to include thousands separators and maintain visual consistency and readability for large numerical values.
## 2026-04-13 - Skip-Link Target Focus Management
**Learning:** When using skip-to-content links that target non-interactive elements (like an `<h2>` heading), keyboard focus might not visually transfer in some browsers, or conversely, it may show an ugly default focus ring when the element receives programmatic focus.
**Action:** Always add `tabindex="-1"` to the skip link's target element so it can programmatically receive focus, and apply CSS (e.g., `[tabindex="-1"]:focus { outline: none; }`) to remove the default focus outline, ensuring a smooth, visually clean transition.
## 2024-11-20 - Chart Tooltip Readability
**Learning:** Default tooltips in Plotly display raw floating-point numbers or large integers without thousands separators, making it difficult for users to read and quickly parse values in interactive dashboards, creating inconsistency with formatted static reports.
**Action:** When customizing Plotly chart tooltips (`hovertemplate`), always use d3-style formatting syntax (e.g., `%{y:,.2f}` or `%{y:,}`) to include thousands separators and maintain visual consistency and readability for large numerical values.
## 2024-05-18 - Document Titles for Screen Readers
**Learning:** Hardcoding generic `<title>` tags like `Water Trends Dashboard` across all visualizations creates a confusing experience for screen reader users navigating between browser tabs, as tabs cannot be distinguished without reading their content.
**Action:** Always extract the specific, dynamic chart title (e.g., from `fig.layout.title.text`) and inject it into the `<title>` tag when generating standalone HTML or reports with Plotly. This ensures clear tab identification for screen readers (WCAG 2.4.2).
## 2024-05-18 - Plotly HTML Export Responsiveness
**Learning:** Default Plotly interactive charts exported to HTML (`fig.to_html()`) do not automatically resize when the browser window or viewport changes, particularly on mobile screens. If a fixed width or height is set in the chart layout, this leads to layout breaks or forces the user to scroll horizontally.
**Action:** When exporting Plotly charts to HTML via `to_html()`, always pass `config={'responsive': True}`. This explicitly enables responsiveness and ensures the chart container adapts fluidly to the screen size.
## 2024-11-20 - HTML Table Data A11y and UX Typography
**Learning:** Pandas `to_html` tables lack proper screen reader associations (`scope="col"`) on header elements, causing cognitive overhead. Additionally, numerical value columns default to left-alignment and proportional fonts, making large numbers difficult to scan and compare across rows.
**Action:** Injected `scope="col"` into generated `<th>` tags for accessibility, and added CSS to right-align the value column (`td:last-child`, `th:last-child`) along with `font-variant-numeric: tabular-nums` to ensure numerical digits align cleanly for UX scanning.
## 2024-10-18 - HTML Table Semantic Captions
**Learning:** When generating HTML tables from Pandas DataFrames (e.g., via `to_html`), missing a semantic `<caption>` tag deprives screen reader users of essential context before they navigate the table data.
**Action:** Always inject a semantic `<caption>` tag into the table output to provide essential context.
## 2024-11-20 - CLI Number Formatting
**Learning:** Users scanning terminal outputs for data tools struggle to read large record counts or metrics without thousands separators. CLI output is an interface too!
**Action:** Always apply `:,` formatting to large numerical outputs in CLI feedback and logs.
## 2024-11-20 - Context-Aware Functional Buttons
**Learning:** Users frequently want to save or print static data summaries, but default browser print dialogs are hidden behind menus, reducing discoverability. However, adding functional buttons (like Print) to static HTML reports clutters the final printed document if not handled correctly.
**Action:** Inject an explicit "Print Report" `<button>` call-to-action in the document header for discoverability, and use `@media print { .print-button { display: none !important; } }` in the CSS to ensure the button is cleanly stripped from the final generated PDF or printed page. Also, ensure the button hides decorative emojis using `aria-hidden="true"`.
## 2026-05-06 - CSS Formatting in Python f-strings
**Learning:** When generating HTML with embedded CSS via Python f-strings, single curly braces for CSS blocks will cause a `KeyError` because Python interprets them as formatting variables.
**Action:** Always escape curly braces by doubling them (`{{` and `}}`) in CSS within Python f-strings to avoid crashes and ensure smooth rendering.

## 2026-05-06 - Smooth Interactive States
**Learning:** Interactive elements in reports (like buttons or hoverable rows) feel jarring if they snap instantly between states. Smooth transitions provide a polished, physical feel that users appreciate.
**Action:** Add `transition: background-color 0.2s ease, transform 0.1s ease;` (or similar) alongside `:hover` and `:active` styles for buttons and table rows in generated reports.
