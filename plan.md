1.  **Understand the Goal**: Identify ONE micro-UX improvement that improves accessibility, usability, or aesthetics based on Palette's guidelines.
2.  **Analyze Current Codebase based on Memory**:
    *   `export_utils.py`: Missing `[title]` selector on `.badge:focus-visible::after`, which creates ghost boxes when the title is temporarily removed by the JS copy script. Also, the JS script modifies the button but leaves the original tooltip showing up if not explicitly managed.
    *   CLI dead ends: Are there CLI dead ends? In `ivi_water/cli.py`, `dashboard`, `fetch-water-data`, `merge-data`, `get-spatial-units`, and `setup_notebooks` commands all have `💡 Tip:` messages.
    *   Empty States with Graceful Visualizations: Checking `visualizer.py`, we see it handles empty DataFrames well by calling `self._create_empty_state_figure(..., message="...")`.
    *   HTML attributes `aria-keyshortcuts`: Added to the print button.
    *   JS in HTML strings `onclick`: Checking the "Copy Table Data" button JS. It uses template literals (backticks).
    *   Let's check `export_utils.py` for `.print-button[title]` vs `.print-button`. Wait! The learning says: "Always include the `[title]` attribute selector (e.g., `.element[title]:focus-visible::after`) on custom CSS tooltip pseudo-elements to ensure they only render when the attribute is present." The current code HAS `.badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after`.
    *   Let's check the copy button logic. In `export_utils.py`:
        ```javascript
        if(this.dataset.active) return;
        this.dataset.active = "1";
        const table = document.querySelector(".summary-table");
        navigator.clipboard.writeText(table.innerText);
        const originalHTML = this.innerHTML;
        const originalTitle = this.title;
        this.removeAttribute("title");
        this.innerHTML = `&lt;span aria-hidden="true"&gt;✅&lt;/span&gt; Copied!`;
        setTimeout(() =&gt; {{
            this.innerHTML = originalHTML;
            this.setAttribute("title", originalTitle);
            delete this.dataset.active;
        }}, 2000);
        ```
        It correctly temporarily removes the `title` attribute.

    *   Wait, what about the CLI setup notebooks command? Memory says: "CLI commands that generate user-editable files (like Jupyter notebooks) can accidentally overwrite user's work... Always add a confirmation prompt (e.g. `click.confirm`) or a `--force` flag before overwriting user-editable files". Let's check `cli.py` for `setup_notebooks`. It takes a `--force` flag. `create_basic_notebook` uses `click.confirm(..., default=False)` if it already exists and `not force`.

    *   Wait! Let's check the tip in CLI for `setup_notebooks`:
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
        The command text `jupyter notebook` is highlighted using `fg="cyan", bold=True`. This follows the learning.

    *   What about other CLI tips?
        ```python
        # get-spatial-units
            + click.style(f"💡 Tip: Use these location IDs with ", fg="yellow")
            + click.style("ivi-water fetch-water-data", fg="cyan", bold=True)
        # fetch-water-data
            + click.style(f"💡 Tip: Visualize these trends with ", fg="yellow")
            + click.style(
                f"ivi-water visualize --data {output_path}", fg="cyan", bold=True
            )
        # merge-data
            + click.style(
                f"💡 Tip: Create a comprehensive dashboard with ", fg="yellow"
            )
            + click.style(
                f"ivi-water dashboard --data {output_path}", fg="cyan", bold=True
            )
        # visualize
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        # dashboard
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        # generate-report
            + click.style(
                f"💡 Tip: Open the generated report file in your browser to view the insights.",
                fg="yellow",
            )
        ```
        The `generate-report` command tip says: "💡 Tip: Open the generated report file in your browser to view the insights.", but it is missing an actionable formatted command! (e.g. it is completely unhighlighted, or lacks a specific file to open). But how to format an open command? On macOS it's `open`, Windows `start`, Linux `xdg-open`. Maybe just highlighting the file path? Wait, there is no actionable command string in a distinct color and weight.
        Let's look for other missing tips. No, the `generate-report` tip doesn't have `fg="cyan", bold=True` for any command. It just outputs text. Wait, is that a problem? The learning says: "When providing actionable commands within CLI '💡 Tip' messages, format the executable command string in a distinct color and weight". If there's no command, maybe that's fine.

    *   Let's check the CLI commands with loading states.
        ```python
        click.echo(
            click.style(
                "⏳ Generating comprehensive dashboard...",
                fg="blue",
            )
        )
        ```
        This matches "always include an explicit visual loading indicator (e.g., `click.echo(click.style("⏳ Fetching...", fg="blue"))`) before the operation".

    *   Let's check empty states in visualizer:
        ```python
        if df.empty:
            return self._create_empty_state_figure(
                title=title,
                message="DataFrame is empty. Please ensure the input data contains records.",
            )
        ```
        This has an actionable message.
        And `_create_empty_state_figure`:
        ```python
        text_color = '#e0e0e0' if self.theme and 'dark' in self.theme.lower() else '#555555'
        ```
        This dynamically sets text color based on theme.

    *   Wait, let's look at `setup_notebooks` Tip again.
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
        Wait, I can add a `💡 Tip:` somewhere? Are there commands missing a tip? `setup_notebooks` has a tip. `generate_report` has a tip.

    *   What about "Add keyboard shortcut hints and SEO metadata to generated HTML reports"?
        `generate_detailed_report` calls `_create_report_index`. Is it generating HTML? No, it generates `.txt`.
        `create_visualization_exports` adds `html_content.replace("<head>", f'<head>\n    <meta name="description" ...`. Yes, SEO is there.

    *   What about "Update Button Contrast for Dark Mode"?
        In `export_utils.py`:
        ```css
        @media (prefers-color-scheme: dark) {{
            ...
            .print-button, .copy-button {{ background-color: #2563eb; }}
            .print-button:hover, .copy-button:hover {{ background-color: #1d4ed8; }}
            ...
        }}
        ```
        This uses `#2563eb` for dark mode buttons.

    *   What about "explicitly define text colors for links to prevent low-contrast defaults"?
        Ah! Memory: "use a deeper shade like #2563eb for the background to ensure sufficient contrast with white text, and explicitly define text colors for links to prevent low-contrast defaults."
        Let's check links in `export_utils.py` dark mode:
        ```css
        .back-to-top {{ color: #60a5fa; }}
        .back-to-top:hover {{ background-color: #1e1e1e; }}
        ```
        Wait, `skip-link` is a link too!
        ```css
                .skip-link {{
                    position: absolute;
                    top: -40px;
                    left: 0;
                    background: #226699;
                    color: white;
                    padding: 8px;
                    z-index: 100;
                    transition: top 0.2s ease;
                    text-decoration: none;
                }}
        ```
        Is it styled for dark mode?
        ```css
                @media (prefers-color-scheme: dark) {{
                    body {{ background-color: #121212; color: #e0e0e0; }}
                    h1 {{ color: #e0e0e0; }}
                    h2 {{ color: #e0e0e0; border-bottom-color: #3b82f6; }}
                    th {{ background-color: #1e1e1e; color: #e0e0e0; border-color: #333; }}
                    td {{ border-color: #333; }}
                    tr:nth-child(even) {{ background-color: #1a1a1a; }}
                    tr:hover {{ background-color: #2c2c2c; }}
                    .summary {{ background-color: #1e1e1e; color: #e0e0e0; border: 1px solid #333; }}
                    .badge {{ background-color: #2c2c2c; color: #e0e0e0; }}
                    .badge:hover, .badge:focus-visible {{ background-color: #3f3f3f; color: #fff; }}
                    .print-button, .copy-button {{ background-color: #2563eb; }}
                    .print-button:hover, .copy-button:hover {{ background-color: #1d4ed8; }}
                    .back-to-top {{ color: #60a5fa; }}
                    .back-to-top:hover {{ background-color: #1e1e1e; }}
                }}
        ```
        `skip-link` has a hardcoded background of `#226699`. In dark mode, `#226699` might be okay, but maybe it should be `#2563eb` for consistency?
