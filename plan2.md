Let's keep looking at `.Jules/palette.md` for clues on what UX I should improve.
"When implementing interactive elements with dynamic text states (like a 'Copy' button updating to 'Copied!'), do not hardcode a static `aria-label` that will mask the updated text. Instead, remove the static `aria-label` if the visible text is sufficient and add `aria-live="polite"` so screen readers accurately announce the state changes."
Let's check the copy button in `export_utils.py` again:
`<button onclick='...' class="copy-button" aria-live="polite" title="Copy Table Data"><span aria-hidden="true">📋</span> Copy Data</button>`
Wait, there is no `aria-label` on the copy button. It just uses text content "📋 Copy Data". So that rule is satisfied.

"When generating HTML reports with custom element backgrounds (like summary boxes, `.summary`, or table headers, `th`), explicitly define the text `color` alongside the `background-color` for each modified element."
Let's check `export_utils.py` for `.summary`:
`.summary {{ background-color: #f8f9fa; color: #333; padding: 15px; border-radius: 5px; }}`
`th {{ background-color: #f2f2f2; color: #333; font-weight: bold; }}`
So that rule is satisfied.

Let's look at UX CHECKS list in the prompt.
ACCESSIBILITY CHECKS:
- Missing ARIA labels, roles, or descriptions
- Insufficient color contrast
- Missing keyboard navigation support (tab order, focus states)
- Images without alt text
- Forms without proper labels or error associations
- Missing focus indicators on interactive elements
- Screen reader unfriendly content
- Missing skip-to-content links

Wait, what about the notebook outputs in `cli.py`?
Notebooks generated don't have UX per se, they are JSON.

Let's review `export_utils.py` again. Is there any missing `aria-label`, `focus-visible`, etc?
`skip-link` has `focus-visible`.
`.badge` has `focus-visible`.
`h1`, `h2` have `tabindex="-1"` and `focus { outline: none; }`? What?
Wait:
`h1[tabindex="-1"]:focus, h2[tabindex="-1"]:focus {{ outline: none; }}`
Why do they have `tabindex="-1"`?
`<h1 id="report-title" tabindex="-1">Water Trends Summary Report</h1>`
`<h2 id="summary-stats-title" tabindex="-1">Summary Statistics</h2>`
`<h2 id="data-overview-title" tabindex="-1">Data Overview</h2>`
This is to allow programmatic focus (e.g. from skip link or anchor links), which is a valid accessibility pattern. Outline none on `-1` is also standard so it doesn't show a focus ring when clicked or scrolled to programmatically.

What about `table-responsive`?
`<div class="table-responsive" tabindex="0" role="region" aria-labelledby="summary-stats-title">`
It has `role="region"`, `aria-labelledby`, `tabindex="0"`. And it has a focus ring:
`.table-responsive:focus-visible {{ outline: 3px solid #ff7f0e; outline-offset: 2px; }}`
This is fully accessible.

What about the `button-group` and buttons?
`<button onclick='...' class="copy-button" aria-live="polite" title="Copy Table Data">`
`<button onClick="window.print()" class="print-button" aria-keyshortcuts="Control+p Meta+p" title="Print Report (Keyboard: Ctrl+P / Cmd+P)">`
Both have focus-visible styles:
`.print-button:focus-visible, .copy-button:focus-visible {{ outline: 3px solid #ff7f0e; outline-offset: 2px; }}`

What about the "Copy Table Data" button tooltips?
`.badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after {{ content: attr(title); ... }}`
Wait! The print button has a title tooltip, but the native title might overlap with the custom CSS one!
Actually, native titles take time to appear. If custom CSS triggers on `:focus-visible`, it works.

Is there any tooltip for the "Back to Top" link?
`<a href="#report-title" class="back-to-top" aria-label="Scroll back to top of the report"><span aria-hidden="true">↑</span> Back to Top</a>`
It has `aria-label`, but does it have a tooltip? No, but the text "Back to Top" is clear.

Wait! The tip for the visualizer HTML generation:
```python
                        # Add accessibility attributes to the graph container with dynamic ARIA label
                        title_text = "Interactive Water Trends Chart"
                        ...
                        html_content = html_content.replace(
                            'class="plotly-graph-div"',
                            f'class="plotly-graph-div" role="region" aria-label="{title_text}" tabindex="0"',
                        )
```
Wait! Is there an issue in Plotly graph generation that needs UX fix?
Plotly graphs are inherently difficult for screen readers, but adding `role="region"`, `aria-label`, and `tabindex="0"` makes the container accessible.

What about the HTML report language?
`<html lang="en">` is present.
`<meta charset="utf-8">` is present.
`<title>` is present.

Wait! Look at `_get_column_tooltip` method in `export_utils.py`!
`"water_area_ha": "Total area of surface water in hectares (1 ha = 10,000 m²)",`
The tooltips are added to the columns badges:
```html
<li class="badge" title="{html.escape(self._get_column_tooltip(str(col)))}" aria-label="Column {html.escape(str(col))}: {html.escape(self._get_column_tooltip(str(col)))}" tabindex="0">{html.escape(str(col))}</li>
```
Wait! The badge has `tabindex="0"`, `title`, and `aria-label`.
When it receives keyboard focus, the custom CSS tooltip uses `content: attr(title)`.
But what about the mouse hover? Does the custom CSS tooltip show on hover?
```css
.badge[title]:focus-visible::after, .print-button[title]:focus-visible::after, .copy-button[title]:focus-visible::after {{ content: attr(title); ...
```
It only triggers on `:focus-visible`. What about `:hover`?
For mouse users, the browser provides a native tooltip from the `title` attribute.

Let's look at `cli.py` again.
Is there any missing loading state?
`get_spatial_units`:
```python
        logger.info(
            f"Fetching spatial units with unit_type='{unit_type}', state='{state}'"
        )
```
Wait, there is no `click.echo(click.style("⏳ Fetching...", fg="blue"))` in `get_spatial_units`!
Let's check `get_spatial_units`:
```python
    try:
        logger.info(
            f"Fetching spatial units with unit_type='{unit_type}', state='{state}'"
        )

        # Validate output filename
        if not output or not isinstance(output, str):
            raise click.ClickException("Output filename must be a non-empty string")

        output_path = Path(ctx.obj["output_dir"]) / output

        # Fetch units
        client = CoREStackClient()
        df = client.get_spatial_units(unit_type=unit_type, state=state)
```
There is no "⏳ Fetching..." message before fetching!
Let's check `fetch_water_data`:
```python
        # Parse locations
        location_list = [loc.strip() for loc in locations.split(",")] if locations else []

        click.echo(
            click.style(
                f"⏳ Fetching water data for {len(location_list)} locations ({start_year}-{end_year})...",
                fg="blue",
            )
        )
```
Ah! `fetch_water_data` HAS the loading message.
Does `merge_data` have it?
```python
        click.echo(
            click.style(
                "⏳ Merging water and NRM data...",
                fg="blue",
            )
        )
```
Does `visualize` have it?
```python
        click.echo(
            click.style(
                f"⏳ Generating {chart_type} chart...",
                fg="blue",
            )
        )
```
Does `dashboard` have it?
```python
        click.echo(
            click.style(
                "⏳ Generating comprehensive dashboard...",
                fg="blue",
            )
        )
```
Does `generate_report` have it?
```python
        click.echo(
            click.style(
                f"⏳ Generating {report_type} report...",
                fg="blue",
            )
        )
```
Does `setup_notebooks` have it?
```python
        click.echo(
            click.style(
                "⏳ Setting up notebooks...",
                fg="blue",
            )
        )
```
Wait! ONLY `get_spatial_units` is missing the `⏳ Fetching spatial units...` loading message!
Let's look at `get_spatial_units`:
```python
        # Validate output filename
        if not output or not isinstance(output, str):
            raise click.ClickException("Output filename must be a non-empty string")

        output_path = Path(ctx.obj["output_dir"]) / output

        # Fetch units
        client = CoREStackClient()
        df = client.get_spatial_units(unit_type=unit_type, state=state)
```
This is a blocking API call. "When executing long-running CLI tasks (like API fetching or data generation), always include an explicit visual loading indicator... to reassure the user that the process has not hung." (Memory: "When executing long-running CLI tasks (like API fetching or data generation), always include an explicit visual loading indicator (e.g., `click.echo(click.style("⏳ Fetching...", fg="blue"))`) before the operation to reassure the user that the process has not hung.")

Is there anything else?
What about empty states in `get_spatial_units`?
```python
        if df.empty:
            click.echo(
                click.style(
                    f"⚠️ No {unit_type}s found"
                    + (f" in {state}" if state else "")
                    + ". Please check your parameters.",
                    fg="yellow",
                )
            )
            return
```
This is fine.

Are there any other UX improvements?
What about the tip in `get_spatial_units`?
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Use these location IDs with ", fg="yellow")
            + click.style("ivi-water fetch-water-data", fg="cyan", bold=True)
        )
```

Wait, what about the tooltip arrow color in dark mode?
In `export_utils.py`:
```css
                .badge[title]:focus-visible::before, .print-button[title]:focus-visible::before, .copy-button[title]:focus-visible::before {{ content: ''; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); border: 6px solid transparent; border-top-color: #333; }}
```
In dark mode:
```css
                    .badge:hover, .badge:focus-visible {{ background-color: #3f3f3f; color: #fff; }}
```
Wait, the tooltip background is `#333` in light mode:
```css
                .badge[title]:focus-visible::after... {{ ... background-color: #333; color: #fff; ... }}
```
But in dark mode, there is NO override for the tooltip background and text color! So the tooltip is `#333` on `#121212` background, which might be okay. Wait, what if we want to invert it, or give it a border? Usually tooltips in dark mode are lighter, e.g., `#e0e0e0` background with `#121212` text.

Wait, looking at `export_utils.py`:
```css
                @media (prefers-color-scheme: dark) {{
                    body {{ background-color: #121212; color: #e0e0e0; }}
                    ...
```
There's no tooltip dark mode styling.

Let's go back to `export_utils.py` and see what else.
Is there an accessibility issue in `export_utils.py` with forms? No forms.
Is there an accessibility issue with the copy button?
"Add tooltip explaining disabled button state." No disabled buttons here.
"Add ARIA label to icon-only button." There are no icon-only buttons (Copy Data and Print Report have text).

What about `click.style` formatting?
Wait, if `get_spatial_units` is missing a loading indicator, adding it is a very small change.
Is there something bigger?

Let's read `cli.py` again.
Is `fetch-water-data` tip missing something?
```python
        click.echo(
            "\n"
            + click.style(f"💡 Tip: Visualize these trends with ", fg="yellow")
            + click.style(
                f"ivi-water visualize --data {output_path}", fg="cyan", bold=True
            )
        )
```
This is good.

Is there any missing "empty state with actionable message" in `cli.py`?
`dashboard`:
```python
        if df.empty:
            click.echo(
                click.style(
                    "⚠️ No data found in the input file. Please ensure the CSV is populated before creating dashboard.",
                    fg="yellow",
                )
            )
            return
```
This is good.

Wait, check the print CSS in `export_utils.py`:
```css
@media print {{ .print-button, .copy-button, .skip-link, .back-to-top {{ display: none !important; }} body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
```
This is perfectly compliant.

What if the enhancement is to add keyboard navigation support (focus states) somewhere it's missing?
Wait, look at `export_utils.py` again:
```css
.back-to-top:focus-visible {{ outline: 3px solid #ff7f0e; outline-offset: 2px; }}
```
All interactive elements seem to have `focus-visible`.

What if the enhancement is about the HTML structure?
```html
                <section aria-labelledby="summary-stats-title">
                    <h2 id="summary-stats-title" tabindex="-1">Summary Statistics</h2>
```
This is accessible.

Wait! What about the `generate-report` command tip?
```python
        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Open the generated report file in your browser to view the insights.",
                fg="yellow",
            )
        )
```
Can we improve this tip to be more actionable? Like giving them the actual command?
But the command to open a file depends on the OS. `open`, `xdg-open`, `start`. It's hard to hardcode one.
However, we can at least colorize the `output_path` so they know *which* file to open.
```python
        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Open the generated report file (", fg="yellow"
            )
            + click.style(f"{output_path}", fg="cyan", bold=True)
            + click.style(") in your browser to view the insights.", fg="yellow")
        )
```
Wait, the missing loading state in `get_spatial_units` is a concrete issue explicitly called out in memory!
"When executing long-running CLI tasks (like API fetching or data generation), always include an explicit visual loading indicator (e.g., `click.echo(click.style("⏳ Fetching...", fg="blue"))`) before the operation to reassure the user that the process has not hung."

Let's check if there are other long-running tasks without loading indicators.
`get_spatial_units` is missing it.
Let's see:
```python
@cli.command()
@click.option(
    "--unit-type",
    required=True,
    type=click.Choice(["village", "micro-watershed", "district"]),
    help="Type of spatial unit",
)
@click.option("--state", help="Filter by state name")
@click.option("--output", default="spatial_units.csv", help="Output filename")
@click.pass_context
def get_spatial_units(
    ctx: click.Context, unit_type: str, state: Optional[str], output: str
) -> None:
    """..."""
    try:
        logger.info(
            f"Fetching spatial units with unit_type='{unit_type}', state='{state}'"
        )

        # ADD LOADING INDICATOR HERE
        state_msg = f" for {state}" if state else ""
        click.echo(
            click.style(
                f"⏳ Fetching {unit_type}s{state_msg} from CoRE Stack...",
                fg="blue",
            )
        )
```
Yes, this is definitely a valid and solid UX enhancement!
Wait, but is there *another* one? What about `cli.py` missing tips entirely on successful operations?
`get_spatial_units` has a tip.
`fetch_water_data` has a tip.
`merge_data` has a tip.
`visualize` has a tip.
`dashboard` has a tip.
`generate_report` has a tip.
`setup_notebooks` has a tip.
All major commands have tips.

Is there any other CLI enhancement?
What about progress bars? Memory says: "When building CLI tools, avoid wrapping blocking or multi-threaded API calls in a progress bar (e.g., `click.progressbar`) if it cannot iteratively track the download progress. Rely on a static loading message (e.g., `click.echo`) instead to prevent the UI from appearing frozen at 0%."
Since there are no `click.progressbar` usage in `cli.py`, this is respected.

Let's reconsider the UX improvement.
"add loading spinner to async submit button" -> For CLI, this means adding `⏳ Fetching...` before long-running tasks.
Let's check if there are other commands in `cli.py` missing it.
I'll create the `get_spatial_units` patch.
