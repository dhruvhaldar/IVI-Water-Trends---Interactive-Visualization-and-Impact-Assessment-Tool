Aha! So `get_spatial_units` *does* have the loading indicator!
Wait, my script output for `test_get_spatial.py` had a different result?
Oh, I see:
```python
        # Fetch spatial units
        click.echo(
            click.style(
                f"⏳ Fetching {unit_type} spatial units...",
                fg="blue",
            )
        )
```
So it ALREADY HAS the loading indicator. My bad.

So what's missing?
Let's check the UX rules from `prompt` and memory again.
- "When generating static HTML reports or charts meant for browser viewing, always include a `<meta name="theme-color" content="[brand-color]">` tag (e.g., `#226699`) in the `<head>` to ensure the mobile browser UI matches the primary brand color and provides a cohesive experience."
I see that this is in `export_utils.py` and `visualizer.py`.

- "When generating static HTML reports, always provide native OS-level Dark Mode support by injecting `@media (prefers-color-scheme: dark)` CSS rules to adjust backgrounds/text, and include `<meta name="color-scheme" content="light dark">` along with responsive `<meta name="theme-color" media="...">` tags to sync the browser UI."
This is also in the code already.

- "When building CLI tools that process datasets, always validate inputs early by checking `if df.empty:` before executing complex logic (like visualization or reporting). If empty, return a styled warning message with actionable advice and an emoji to prevent confusing downstream failures."
Let's check `visualize`, `dashboard`, `generate_report`, etc.
They all have:
```python
        if df.empty:
            click.echo(
                click.style(
                    "⚠️ No data found in the input file. Please ensure the CSV is populated before visualizing.",
                    fg="yellow",
                )
            )
            return
```
This is fully compliant.

- "When adding text annotations to Plotly charts (e.g., empty state messages), explicitly define text color dynamically based on the active theme (e.g., `#555555` for light themes, `#e0e0e0` for dark themes) to ensure WCAG AA contrast requirements are met across all environments, rather than relying on inherited defaults or hardcoding a single color."
Let's check `visualizer.py`:
```python
        # Explicitly define text color dynamically based on the active theme
        # to ensure WCAG AA contrast requirements are met across all environments.
        text_color = '#e0e0e0' if self.theme and 'dark' in self.theme.lower() else '#555555'
```
Wait! Does it use `text_color` in the annotation? Let's check `_create_empty_state_figure`.
