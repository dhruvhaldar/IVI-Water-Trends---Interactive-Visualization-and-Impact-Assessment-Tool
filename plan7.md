Okay, I see the loading indicator in `get_spatial_units` is there.

What about:
`"When providing actionable commands within CLI '💡 Tip' messages, format the executable command string in a distinct color and weight (e.g., `fg='cyan', bold=True` with `click.style`) to ensure it stands out for easy copy-pasting."`

Let's check `cli.py` line 894 in `generate_report` command:
```python
        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Open the generated report file in your browser to view the insights.",
                fg="yellow",
            )
        )
```

Is there any other Tip that misses it?
Line 324: `get_spatial_units`
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Use these location IDs with ", fg="yellow")
            + click.style("ivi-water fetch-water-data", fg="cyan", bold=True)
        )
```
Line 564: `fetch_water_data`
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Visualize these trends with ", fg="yellow")
            + click.style(
                f"ivi-water visualize --data {output_path}", fg="cyan", bold=True
            )
        )
```
Line 654: `merge_data`
```python
        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Create a comprehensive dashboard with ", fg="yellow"
            )
            + click.style(
                f"ivi-water dashboard --data {output_path}", fg="cyan", bold=True
            )
        )
```
Line 741: `visualize`
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        )
```
Line 821: `dashboard`
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        )
```
Line 967: `setup_notebooks`
```python
        if created:
            click.echo(
                "\n"
                + click.style(
                    f"💡 Tip: Start Jupyter notebook to begin your analysis. ",
                    fg="yellow",
                )
                + click.style("jupyter notebook", fg="cyan", bold=True)
            )
```

All of them have `click.style(..., fg="cyan", bold=True)` EXCEPT `generate_report`.

Is there an actionable command string to open the file?
I can just format the filename itself, e.g. `f"open {output_path}"`. Wait, if we format it as an actionable command, it will follow the rule. Or maybe there's no actionable command. But if I change it to `click.style(f"open {output_path}", fg="cyan", bold=True)`, that's actionable on Mac, though not Windows. But wait! I can also just format the path.

What about `export_utils.py`?
Is there a custom CSS tooltip that's missing the `[title]` attribute?
```css
.badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after
```
No, they all have `[title]`.

Wait, what about the tooltip styling for dark mode in `export_utils.py`?
In `.badge[title]:focus-visible::after`, `background-color: #333; color: #fff`.
In dark mode, it stays `#333`, which is very dark on a `#121212` background. This means the tooltip has very low contrast against the background in dark mode. The border arrow (`border-top-color: #333`) also blends into the dark background.
Wait! `export_utils.py` has no dark mode styles for `.badge[title]:focus-visible::after`!
Let's see if we can add dark mode tooltip styling.

```css
                @media (prefers-color-scheme: dark) {{
                    body {{ background-color: #121212; color: #e0e0e0; }}
                    ...
                    .badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after {{ background-color: #e0e0e0; color: #121212; }}
                    .badge[title]:focus-visible::before, .print-button[title]:focus-visible::before, .copy-button[title]:focus-visible::before {{ border-top-color: #e0e0e0; }}
                }}
```

Wait, is there any other UX issue?
Let's review memory again.
- "When generating static HTML reports or charts meant for browser viewing, always include a `<meta name="theme-color" content="[brand-color]">` tag (e.g., `#226699`) in the `<head>` to ensure the mobile browser UI matches the primary brand color and provides a cohesive experience."
In `export_utils.py` `_create_html_report`:
```html
            <meta name="theme-color" content="#226699" media="(prefers-color-scheme: light)">
            <meta name="theme-color" content="#121212" media="(prefers-color-scheme: dark)">
```

- "When generating static HTML reports, always provide native OS-level Dark Mode support by injecting `@media (prefers-color-scheme: dark)` CSS rules to adjust backgrounds/text, and include `<meta name="color-scheme" content="light dark">` along with responsive `<meta name="theme-color" media="...">` tags to sync the browser UI."
This is present.

- "When building CLI tools that process datasets, always validate inputs early by checking `if df.empty:` before executing complex logic (like visualization or reporting). If empty, return a styled warning message with actionable advice and an emoji to prevent confusing downstream failures."
This is present.

- "When styling elements for dark mode, avoid default bright blues (like `#3b82f6`) as they often fail WCAG contrast ratios against white text. Use deeper shades (like `#2563eb`) for button backgrounds and lighter shades (like `#60a5fa`) for text links against dark backgrounds to ensure accessibility."
In `export_utils.py`:
```css
                    .print-button, .copy-button {{ background-color: #2563eb; }}
                    .print-button:hover, .copy-button:hover {{ background-color: #1d4ed8; }}
                    .back-to-top {{ color: #60a5fa; }}
```
Wait! What about the skip link?
```css
                .skip-link {{
                    ...
                    background: #226699;
                    color: white;
                    ...
                }}
```
Is the skip-link updated in dark mode?
No!
```css
                @media (prefers-color-scheme: dark) {{
                    body {{ background-color: #121212; color: #e0e0e0; }}
                    h1 {{ color: #e0e0e0; }}
                    h2 {{ color: #e0e0e0; border-bottom-color: #3b82f6; }}
```
Wait, `h2` uses `#3b82f6` for `border-bottom-color`!
Memory: "When styling elements for dark mode, avoid default bright blues (like `#3b82f6`) as they often fail WCAG contrast ratios against white text. Use deeper shades (like `#2563eb`) for button backgrounds..."
But for a border, `#3b82f6` might be okay because it's not text.
Still, the skip link is `#226699` (dark blue) which is not explicitly set in dark mode. `#2563eb` could be used. But it's not a bright blue.

Wait! Is there an empty state for visualizations that's missing the theme logic or actionable advice?
In `visualizer.py`:
`def _create_empty_state_figure(...)`
```python
        # Explicitly define text color dynamically based on the active theme
        # to ensure WCAG AA contrast requirements are met across all environments.
        text_color = '#e0e0e0' if self.theme and 'dark' in self.theme.lower() else '#555555'
```
This is present.

Wait! What about `cli.py` dead ends?
"To prevent dead-end experiences in CLI tools and improve discoverability, always append context-aware "💡 Tip: [Actionable advice]" messages to the successful output of major commands to guide users logically through the workflow."
Let's see if there are any commands without tips.
- `setup_notebooks`: Yes.
- `generate_report`: Yes, but it says "Open the generated report file in your browser".
- `dashboard`: Yes, suggests `generate-report`.
- `visualize`: Yes, suggests `generate-report`.
- `merge_data`: Yes, suggests `dashboard`.
- `fetch_water_data`: Yes, suggests `visualize`.
- `get_spatial_units`: Yes, suggests `fetch_water_data`.

What about `cli.py` confirmation?
"When building CLI tools that generate user-editable artifacts (like Jupyter notebooks), always implement a confirmation prompt (e.g., using `click.confirm`) or require a `--force` flag before overwriting existing files to prevent accidental data loss."
In `cli.py` `create_basic_notebook`:
```python
    if notebook_path.exists() and not force:
        if not click.confirm(
            click.style(
                f"⚠️  Notebook {notebook_path.name} already exists. Overwrite and lose any changes?",
                fg="yellow",
            ),
            default=False,
        ):
            return False
```
This is present.

Is there any missing "aria-keyshortcuts"?
`export_utils.py` line 700:
`<button onClick="window.print()" class="print-button" aria-keyshortcuts="Control+p Meta+p" title="Print Report (Keyboard: Ctrl+P / Cmd+P)">`
This is present.

Is there a missing line height?
`body {{ font-family: ...; line-height: 1.6; }}`
This is present.

Is there missing print color adjust?
`@media print {{ ... body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}`
This is present.

Is there any issue with `title` overlapping the dynamic button text?
"When updating the text of an interactive element dynamically (e.g., a button changing to 'Copied!'), temporarily remove its title attribute (tooltip) during the active state to prevent a stale or conflicting tooltip from displaying, and restore it once the state resets."
`export_utils.py` line 695:
```javascript
const originalTitle = this.title; this.removeAttribute("title"); ... this.setAttribute("title", originalTitle);
```
This is present.

Is there any missing `aria-live`?
```html
class="copy-button" aria-live="polite"
```
This is present.

What about:
"Since native title tooltips do not reliably display on keyboard focus, implement custom CSS tooltips (e.g., using `:focus-visible::after { content: attr(title); }` and `position: relative;`) on focusable elements to ensure tooltip text is accessible to sighted keyboard users."
This is present in `export_utils.py` for `.badge`, `.print-button`, `.copy-button`.

Wait! Check the tooltip definition in `export_utils.py`:
```css
.badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after {{ content: attr(title); position: absolute; bottom: 130%; left: 50%; transform: translateX(-50%); background-color: #333; color: #fff; padding: 6px 10px; border-radius: 4px; white-space: normal; width: max-content; max-width: 250px; z-index: 10; pointer-events: none; font-size: 12px; line-height: 1.4; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
.badge[title]:focus-visible::before, .print-button[title]:focus-visible::before, .copy-button[title]:focus-visible::before {{ content: ''; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); border: 6px solid transparent; border-top-color: #333; }}
```
Wait! What happens if I focus the element, the tooltip shows. What if I mouse hover it?
The mouse hover triggers the native browser tooltip because it has a `title` attribute.
Is that a problem? No, the rule just says "ensure tooltip text is accessible to sighted keyboard users."

What about "Add ARIA label to icon-only button."
Is there any icon-only button? No, they all have text.
Wait, let's look at `visualizer.py` or other files. Is there an icon-only button? No HTML besides `export_utils.py`.

What about "Add loading spinner to async submit button"?
Wait. "Add loading spinner to async submit button". There is no async submit button in HTML. The CLI loading state is present.

What about "Forms without proper labels or error associations"?
There are no forms.

What about "Missing 'required' indicators on form fields"?
There are no forms.

What about "Missing empty state with helpful call-to-action"?
Wait, `visualizer.py` has an empty state. It says: "DataFrame is empty. Please ensure the input data contains records."

Let's rethink: The goal is to "find and implement ONE micro-UX improvement that makes the interface more intuitive, accessible, or pleasant to use."
"keep changes under 50 lines"

What if the enhancement is simply to fix the tooltip contrast in Dark Mode?
In dark mode, the tooltip (`#333` on `#121212`) is nearly invisible. Adding dark mode styles for the tooltip pseudo-elements is definitely an accessibility improvement.
```css
                @media (prefers-color-scheme: dark) {{
                    ...
                    .badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after {{ background-color: #e0e0e0; color: #121212; }}
                    .badge[title]:focus-visible::before, .print-button[title]:focus-visible::before, .copy-button[title]:focus-visible::before {{ border-top-color: #e0e0e0; }}
                }}
```

Are there any other opportunities?
What about the skip link in dark mode?
It's `#226699`. A bit dark, maybe it should be `#2563eb` like the buttons?
```css
                    .skip-link {{ background-color: #2563eb; }}
```

What about the `button-group`?
```css
.button-group {{ display: flex; gap: 10px; }}
```

What about the `generate_report` CLI tip?
```python
        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Open the generated report file ", fg="yellow"
            )
            + click.style(f"{output_path}", fg="cyan", bold=True)
            + click.style(" in your browser to view the insights.", fg="yellow")
        )
```
This also fixes the actionable command text color format.

Wait, check this memory item:
"When updating the text of an interactive element dynamically (e.g., a button changing to 'Copied!'), temporarily remove its `title` attribute (tooltip) during the active state to prevent a stale or conflicting tooltip from displaying, and restore it once the state resets."
This is exactly what the copy button JS is doing!
```javascript
const originalTitle = this.title; this.removeAttribute("title"); ... this.setAttribute("title", originalTitle);
```

Check this memory item:
"When generating HTML reports with custom element backgrounds (like summary boxes, `.summary`, or table headers, `th`), explicitly define the text `color` alongside the `background-color` for each modified element. This prevents critical contrast/readability issues for users whose browser or OS is configured for dark mode."
```css
.summary {{ background-color: #f8f9fa; color: #333; padding: 15px; border-radius: 5px; }}
```
This is correct.

Check this memory item:
"When implementing inline JavaScript actions with temporary visual feedback (like a 'Copy Data' button updating its innerHTML), prevent double-click race conditions by tracking state using a dataset flag (e.g., `if(this.dataset.active) return; this.dataset.active = '1';`) and cleaning it up after the timeout or operation completes."
This is correct in `export_utils.py`!

Wait, is there anything WRONG in `export_utils.py` right now?
Let's look at:
```css
                    h2 {{ color: #e0e0e0; border-bottom-color: #3b82f6; }}
```
"When styling elements for dark mode, avoid default bright blues (like `#3b82f6`) as they often fail WCAG contrast ratios against white text. Use deeper shades (like `#2563eb`) for button backgrounds and lighter shades (like `#60a5fa`) for text links against dark backgrounds to ensure accessibility."
The `h2` border in dark mode is `#3b82f6`. Maybe it should be `#60a5fa` or `#2563eb`? Wait, `#3b82f6` is specifically called out as a color to avoid for text contrast, but here it's a border. Still, updating it might be the point.

Let's look at `visualizer.py`.
```python
        html_content = html_content.replace(
            '<head>',
            f'<head>\n    <meta name="description" content="Interactive visualization of water trends.">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta name="color-scheme" content="light dark">\n    <meta name="theme-color" content="#226699" media="(prefers-color-scheme: light)">\n    <meta name="theme-color" content="#121212" media="(prefers-color-scheme: dark)">\n    <title>{title_text}</title>\n    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌊</text></svg>">\n    <style>.plotly-graph-div:focus-visible {{ outline: 3px solid #ff7f0e; outline-offset: 2px; border-radius: 4px; }}\n    @media (prefers-color-scheme: dark) {{ body {{ background-color: #121212; color: #e0e0e0; }} }}\n    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto !important; }} * {{ transition: none !important; transform: none !important; }} }}</style>',
        )
```
This all looks perfect.
Wait! What if we add a "Copy Chart Image" button to the visualizer? No, Plotly already has a toolbar.

Wait, check `cli.py` again.
Is there any command that lacks a tip?
No.
Is there any command that lacks a loading indicator?
No.

What about `cli.py` tip formatting:
```python
            + click.style(f"💡 Tip: Use these location IDs with ", fg="yellow")
            + click.style("ivi-water fetch-water-data", fg="cyan", bold=True)
```
This is fully compliant with the memory: "format the executable command string in a distinct color and weight (e.g., `fg='cyan', bold=True` with `click.style`) to ensure it stands out for easy copy-pasting."

Wait! Let's check the tip in `visualize` and `dashboard`:
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        )
```
This is fully compliant.

So what's the missing UX fix?
Let's re-read the PR presentation requirements:
"Title: '🎨 Palette: [UX improvement]'
Description with:
💡 What: The UX enhancement added
🎯 Why: The user problem it solves
📸 Before/After: Screenshots if visual change
♿ Accessibility: Any a11y improvements made"

Maybe the focus ring on buttons in `export_utils.py`?
`.print-button:focus-visible, .copy-button:focus-visible {{ outline: 3px solid #ff7f0e; outline-offset: 2px; }}`
This is fully compliant.

Maybe there's a missing `cursor: pointer` on some element?
The buttons have `cursor: pointer`.

Maybe I can improve the "generate_report" CLI tip to include an actionable path?
Or maybe I can improve the Dark Mode Custom Tooltips? The custom CSS tooltips (`#333` background) are basically invisible on the `#121212` background in dark mode. This directly impacts the accessibility of sighted keyboard users in dark mode. They tab to the button, the tooltip pops up, but it's black-on-black, so they can't read it.
This perfectly fits the "micro-UX improvement that improves accessibility" requirement!

Let me fix the dark mode CSS tooltip in `export_utils.py`, AND fix the CLI tip text formatting in `cli.py`. Both are very small fixes.
