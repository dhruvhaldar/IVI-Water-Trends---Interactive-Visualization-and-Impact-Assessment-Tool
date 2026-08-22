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
## 2024-05-09 - PDF Report Table Contrast Ratio
**Learning:** The default `colors.grey` background with `colors.whitesmoke` text in ReportLab tables produces a contrast ratio of ~2.5:1, which fails WCAG AA standards (4.5:1 required) and makes the PDF headers hard to read.
**Action:** Replaced `colors.grey` with a brand-aligned `colors.HexColor('#226699')` to achieve a 7.8:1 contrast ratio, passing WCAG AAA and improving readability for all users.

## 2024-05-24 - Smooth Scrolling & Back-to-Top in HTML Reports
**Learning:** Generated HTML reports with large data tables can trap keyboard users at the bottom of the document. Standard anchor links without smooth scrolling can be disorienting.
**Action:** Always include a "Back to Top" link with `scroll-behavior: smooth` in generated long-form HTML documents to ensure keyboard users can easily return to the main navigation/header, and use `@media print` to hide it.

## 2024-05-11 - Table Row Headers Accessibility
**Learning:** By default, Pandas `to_html()` generates data rows with `<td>` tags for the first column. This makes it difficult for screen readers to properly associate row data with their corresponding row headers, creating cognitive overhead for users relying on assistive technologies.
**Action:** Always ensure that row headers in HTML tables (the first column indicating what the row represents) use `<th scope="row">` instead of `<td>` to properly define their semantic role and improve accessibility for screen readers.
## 2026-05-18 - Keyboard Shortcut Hints for Print Actions\n**Learning:** Functional actions like 'Print' have standard browser keyboard shortcuts (Ctrl+P / Cmd+P), but these are not always obvious to users when a custom button is provided in the UI.\n**Action:** When adding functional buttons (e.g., 'Print Report') to generated HTML interfaces, always include a `title` attribute with the corresponding keyboard shortcut (e.g., `title="Print Report (Keyboard: Ctrl+P / Cmd+P)"`) to act as a tooltip and improve discoverability for power users and keyboard navigators.

## 2024-05-24 - HTML Report Base Colors vs OS Dark Mode
**Learning:** If an HTML report specifies background colors for some elements (like tables or summary boxes) but leaves the `body` background transparent, users whose OS or browser defaults to dark mode might see light text rendered against those light hardcoded backgrounds, making the text invisible.
**Action:** When generating static HTML reports with custom element backgrounds, always explicitly define `color` and `background-color` on the `body` tag to ensure proper contrast regardless of user browser preferences.
## 2026-05-15 - Enhancing Quick-Scannable Summary Metrics
**Learning:** Raw `<p>` tags for important standalone stats (like "Total Records") blend in with surrounding text, making them hard to scan quickly in generated reports.
**Action:** Always wrap key metrics in `<strong>` and utilize existing callout containers (like the `.summary` class) to elevate their visual hierarchy and scannability, avoiding plain paragraph text for high-value data.

## 2024-05-24 - Delightful Tab Icons for Static HTML Exports
**Learning:** Generated standalone HTML reports and charts often lack favicons, resulting in generic browser default icons (like a blank page). This makes it difficult for users to identify the report tab among many open tabs, reducing usability.
**Action:** Always inject a lightweight SVG data-URI favicon (e.g., `<link rel="icon" href="data:image/svg+xml,...">`) using a relevant emoji into the `<head>` of generated HTML files to provide an instantly recognizable and delightful touch without needing external assets.

## 2024-05-24 - Aria-Hidden for Decorative Text Symbols
**Learning:** Using text characters as decorative icons (like "↑" for a back-to-top link) causes screen readers to announce them literally (e.g., "Upwards arrow Back to Top"), creating a confusing and noisy experience for visually impaired users.
**Action:** Always wrap decorative text symbols in `<span aria-hidden="true">` to ensure they are ignored by screen readers while remaining visible to sighted users.
## 2024-05-19 - Dark Mode Accessibility Fix in HTML Reports
**Learning:** Hardcoded white backgrounds (`background-color: #fff;`) with unstyled `color` properties on child elements (or default light text) can lead to invisible text for users with OS/browser-level Dark Mode enabled.
**Action:** When setting specific background colors (like white or light gray) on elements like `.summary` or the `body` tag, always explicitly define the text `color` to ensure sufficient contrast regardless of the user's system preferences.
## 2026-05-24 - HTML Table Data A11y and UX Typography (Multiple Columns)
**Learning:** Hardcoding `:last-child` for right-aligning numerical table columns only formats the final column, leaving other numerical columns (like mean, min, std in a summary table) left-aligned and hard to scan.
**Action:** Use `:not(:first-child)` to apply tabular formatting and right-alignment to all data columns in summary tables, assuming the first column is the row header.

## 2024-05-21 - CLI Success Feedback Enhancements
**Learning:** Text-only or plainly formatted CLI feedback (e.g., using a plain "✓") can easily get lost in dense terminal output, making it harder for users to quickly scan for successful task completions. Consistent color and highly visible emojis dramatically improve the CLI's scannability.
**Action:** Consistently use the `✅` emoji and explicit `click.style(..., fg="green")` to style success messages in CLI tools to ensure they stand out clearly against standard terminal text.

## 2024-11-20 - CLI Tabular Data Formatting and Scannability
**Learning:** Printing tabular data to the CLI without column headers forces users to guess what each value represents, causing cognitive friction. Additionally, secondary text (like "... and 40 more records") can distract from the main data if not visually de-emphasized.
**Action:** Always include explicit tabular headers styled distinctly (e.g., cyan color) when printing table rows in the CLI, and use `click.style(..., dim=True)` to de-emphasize secondary or trailing text to keep user focus on the primary data.
## 2024-05-15 - Empty State Actionable Advice
**Learning:** When displaying empty state messages like 'No data found' or 'No significant insights detected', users benefit greatly from actionable advice to recover.
**Action:** Always include a clear warning emoji (like ⚠️) and provide actionable advice (e.g., 'Try expanding your date range') to guide the user on how to recover.
## 2024-05-25 - HTML Report Reduced Motion Accessibility
**Learning:** Generated HTML reports use `scroll-behavior: smooth` for back-to-top links and CSS `transition`/`transform` effects on interactive elements like buttons and rows. For users with vestibular disorders, these motion effects can trigger dizziness or nausea.
**Action:** Always inject an `@media (prefers-reduced-motion: reduce)` CSS block into HTML report templates that forces `scroll-behavior: auto !important` on the `html` element and disables all transitions (`* { transition: none !important; transform: none !important; }`). This ensures the UI respects OS-level accessibility preferences while retaining visual polish for other users.
## 2024-06-02 - Keyboard Accessible Tooltips for Metadata
**Learning:** Adding descriptive `title` tooltips to metadata elements (like dataset column badges) provides excellent contextual help, but is inaccessible to keyboard users by default. Standard non-interactive elements (like `<li>` or `<span>`) do not receive focus, preventing keyboard users from seeing the native browser tooltip.
**Action:** When adding tooltips to non-interactive informative elements, always add `tabindex="0"` to allow keyboard focus, and apply a clear `:focus-visible` outline alongside `cursor: help` to indicate interactivity to all users.

## 2026-05-25 - HTML Report Mobile Browser UI Cohesion
**Learning:** By default, mobile browser UIs (like the address bar and status bar) use a default system color, which can look disconnected from a generated HTML report or chart that has a strong brand identity.
**Action:** When generating static HTML reports or interactive charts meant for browser viewing, always inject a `<meta name="theme-color" content="[brand-color]">` tag (e.g., `#226699`) into the `<head>` to ensure the mobile browser UI matches the primary brand color, providing a seamless and polished experience.
## 2026-05-30 - Empty States with Actionable Advice
**Learning:** When executing CLI commands that expect data inputs (like generating reports or visualizations), users are easily confused if the process fails silently or errors out when the input CSV is empty.
**Action:** Always include an empty state check (e.g., `if df.empty:`) and provide clear, actionable advice with a warning emoji (e.g., '⚠️ No data found in the input file. Please ensure the CSV is populated before visualizing.') to guide users appropriately.
## 2026-06-10 - Add explicit loading states to long-running CLI tasks
**Learning:** Users often assume a CLI tool has hung if there is no immediate visual feedback during network requests or expensive data generation operations.
**Action:** Always include a visual loading indicator (e.g., `click.echo(click.style("⏳ Fetching...", fg="blue"))`) before long-running tasks to reassure users that the process is working.

## 2024-05-25 - Native Dark Mode Support in Static Reports
**Learning:** Static HTML reports lacking native OS-level Dark Mode support (`@media (prefers-color-scheme: dark)`) create a jarring, overly bright experience for users whose systems are set to dark mode. Without matching `<meta name="color-scheme" content="light dark">` and `<meta name="theme-color" media="...">` tags, the browser UI and document canvas will clash with user preferences.
**Action:** Always inject `@media (prefers-color-scheme: dark)` CSS to adjust backgrounds, text colors, and component borders appropriately, and ensure the corresponding responsive `<meta name="theme-color">` tags are present in the `<head>` of generated HTML reports.

## 2026-06-18 - Explicit Loading States in Long-Running CLI Tasks
**Learning:** Users often assume a CLI tool has hung if there is no immediate visual feedback during file I/O operations or data processing steps.
**Action:** Always include a visual loading indicator (e.g., `click.echo(click.style("⏳ Action...", fg="blue"))`) before potentially slow tasks to reassure users that the process is working.

## 2024-06-19 - Dashboard empty state feedback
**Learning:** Running the `dashboard` CLI command with non-existent location IDs causes a crash deep within the visualization logic, providing a confusing traceback.
**Action:** Always validate that filtered datasets are not empty before passing them to visualization functions, and provide a clear warning emoji (⚠️) with actionable advice.

## 2026-07-03 - HTML Semantic `<time>` tags and reliable Tooltip `aria-label`
**Learning:** Using `title` attributes on non-interactive elements like `<li>` lists for tooltips works fine visually when focused with `tabindex="0"`, but screen readers can be extremely inconsistent about announcing these `title` tooltips. Furthermore, string-based timestamps lack structural meaning for assistive technology and machine parsing.
**Action:** When adding interactive tooltips via `title` to custom UI components, always provide a fallback `aria-label` (e.g., `aria-label="Column {col}: {tooltip}"`) to guarantee screen readers voice the description. Always wrap rendered timestamps in semantic `<time datetime="{isoformat}">` tags to improve structural document semantics.

## 2024-07-04 - Optimal Line Height for Text Readability
**Learning:** Browsers typically default to a line-height of roughly 1.2. In data-dense HTML reports, this makes reading paragraphs or lists fatiguing and violates WCAG AAA accessibility recommendations (which suggest a line spacing of at least 1.5).
**Action:** When generating static HTML reports with custom CSS, always explicitly set `line-height: 1.5` (or `1.6`) on the `body` element to ensure long text blocks and summaries remain highly legible for all users.

## 2026-07-05 - Preserving HTML Report Visuals in Print
**Learning:** Browsers strip background colors by default when printing HTML pages to save ink. In data reports, this destroys the visual hierarchy of summary boxes (`.summary`) and table headers (`th`), making the printed version harder to read and reducing contrast.
**Action:** Always include `-webkit-print-color-adjust: exact; print-color-adjust: exact;` in the `@media print` CSS block of generated static HTML reports to ensure background colors are preserved in the final printed document or PDF export.

## 2026-07-09 - CLI Next-Step Contextual Tips
**Learning:** CLI tools often create dead-end experiences after a successful command execution. Without clear guidance on what to do next, users (especially beginners) struggle to discover downstream commands and the full workflow capability.
**Action:** Always append context-aware "💡 Tip: [Actionable advice]" messages to the successful output of major CLI commands (e.g., suggesting visualization after data fetching) to improve discoverability and guide the user logically through the tool's intended workflow.

## 2026-07-13 - Inline JS syntax formatting in Python HTML generation
**Learning:** When injecting inline JavaScript that contains HTML strings into an HTML attribute via Python f-strings (e.g., `onclick="..."`), you must avoid using backslash-escaped single quotes (`\'`) and be careful with HTML entities like `&quot;`. When the browser parses the HTML document, it decodes HTML entities *before* the JavaScript engine evaluates the inline script. The `&quot;` entities decode into literal double-quotes (`"`), prematurely closing the JavaScript string literal and resulting in a `SyntaxError: missing ) after argument list`. A safer alternative is to use JS template literals (backticks `\``) for inner HTML strings inside the inline Javascript, keeping the HTML double quotes intact inside them, e.g. ``this.innerHTML = `<span aria-hidden="true">✅</span> Copied!`;``. Remember to escape inner curly braces from python string formatting.
**Action:** Be mindful of how HTML entity decoding affects inline JS in HTML attributes. Use template literals (backticks) for inner strings to maintain correct syntax and avoid entity-decoding-related syntax errors. Always test the output HTML string directly to ensure no syntax errors are introduced.

## 2024-11-20 - CLI confirmation for editable artifacts
**Learning:** CLI commands that generate user-editable files (like Jupyter notebooks) can accidentally overwrite user's work, causing data loss and poor UX.
**Action:** Always add a confirmation prompt (e.g. `click.confirm`) or a `--force` flag before overwriting user-editable files to prevent accidental data loss.

## 2024-06-25 - Empty States with Graceful Visualizations
**Learning:** When building data visualization functions (e.g., returning Plotly `go.Figure` objects) that accept Pandas DataFrames, users and downstream dashboards are easily broken if the process fails silently or raises exceptions like `ValueError` when the input DataFrame is empty.
**Action:** Always handle the `df.empty` case by returning a graceful "empty state" figure (e.g., with hidden axes and an actionable annotation/message like "⚠️ No valid data to display") rather than throwing a `ValueError` or failing silently. This prevents unexpected application or dashboard crashes and gives users clear feedback.
## 2026-07-23 - Add actionable advice to chart empty states
**Learning:** When data visualizations fail to render because of empty datasets, the default 'DataFrame is empty' message leaves users confused about how to fix the issue.
**Action:** Always handle `df.empty` cases in visualization functions by returning a graceful empty state figure with an actionable message (e.g., 'DataFrame is empty. Please ensure the input data contains records.') instead of a generic error.
## 2024-05-20 - Actionable Empty States
**Learning:** When users encounter empty states, just telling them it's empty is a dead end. Providing actionable advice like 'Please check your data source' drastically improves UX.
**Action:** Always include actionable guidance when showing an empty state message.
## 2025-05-20 - Dynamic Button State Accessibility
**Learning:** Hardcoding `aria-label` on interactive elements like a 'Copy' button prevents screen readers from announcing subsequent DOM changes. When inline JS updates the button text to 'Copied!', the screen reader still reads the static `aria-label`.
**Action:** Remove the static `aria-label` when the button's visible text provides sufficient context, and add `aria-live="polite"` so screen readers accurately announce state changes (like success messages) to non-visual users.
## 2026-08-01 - Dynamic Tooltips on Interactive State Changes
**Learning:** When using an inline JavaScript `onclick` handler to update the visible text of a button (e.g., from 'Copy Data' to 'Copied!'), any hardcoded `title` attribute acts as a tooltip that remains unchanged. This causes a confusing UX where the tooltip (e.g., 'Copy Table Data') persists even after the button indicates success.
**Action:** When updating the text of an interactive element dynamically, temporarily remove its `title` attribute (tooltip) during the active state using `removeAttribute("title")` (while preserving the original value), and restore it using `setAttribute` once the state resets (e.g., after a `setTimeout`), ensuring clear feedback for both screen readers (via `aria-live`) and visual users.
## 2024-05-31 - Custom Tooltips for Keyboard Focus
**Learning:** Native `title` tooltips typically do not appear when an element receives keyboard focus, making them inaccessible to sighted keyboard users.
**Action:** When creating custom tooltips using `title` on focusable elements (like `.badge`), always implement CSS pseudo-elements (e.g., `::after` with `content: attr(title);` and `::before` for the arrow) triggered by `:focus-visible` to ensure the tooltip text is visible during keyboard navigation. Remember to set `position: relative;` on the parent element.
## 2026-08-02 - Custom CSS Tooltips Ghost Boxes
**Learning:** Custom CSS tooltips using `content: attr(title)` will render empty dark boxes (ghost tooltips) when inline JavaScript temporarily removes the `title` attribute for visual feedback states.
**Action:** Always include the `[title]` attribute selector (e.g., `.element[title]:focus-visible::after`) on custom CSS tooltip pseudo-elements to ensure they only render when the attribute is present.
## 2026-08-06 - Add explicit keyboard shortcut hints and SEO metadata to generated HTML reports
**Learning:** While tooltips provide visual cues for keyboard shortcuts (like Ctrl+P for Print), screen reader users may miss them if they are only in the `title` attribute. Additionally, generated HTML files often lack basic SEO and context metadata.
**Action:** Always add `aria-keyshortcuts` to buttons with keyboard shortcuts, and inject a `<meta name="description">` tag into the `<head>` of generated standalone HTML files.
## 2026-08-07 - Add explicit SEO metadata to generated HTML reports
**Learning:** Generated HTML files often lack basic SEO and context metadata like descriptions, which are helpful for screen readers to provide page context.
**Action:** Always inject a `<meta name="description">` tag into the `<head>` of generated standalone HTML files.
## 2026-08-13 - Update Button Contrast for Dark Mode
**Learning:** Default Tailwind-style blue colors like #3b82f6 often fail WCAG contrast ratios against white text in dark mode environments, creating accessibility barriers for visually impaired users.
**Action:** When styling buttons for dark mode, use a deeper shade like #2563eb for the background to ensure sufficient contrast with white text, and explicitly define text colors for links to prevent low-contrast defaults.
## 2024-08-22 - Dynamic Empty State Annotation Colors
**Learning:** When adding text annotations to Plotly charts (e.g., empty state messages), hardcoding a single color like `#555555` often fails WCAG contrast requirements when the chart is rendered in a dark theme.
**Action:** Explicitly define text color dynamically based on the active theme (e.g., `#555555` for light themes, `#e0e0e0` for dark themes) to ensure WCAG AA contrast requirements are met across all environments.
