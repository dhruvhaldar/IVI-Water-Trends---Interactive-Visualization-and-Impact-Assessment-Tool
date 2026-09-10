Yes! `_create_empty_state_figure` defines `text_color`:
```python
        # Explicitly define text color dynamically based on the active theme
        # to ensure WCAG AA contrast requirements are met across all environments.
        text_color = '#e0e0e0' if self.theme and 'dark' in self.theme.lower() else '#555555'
```
Wait! Does it actually USE `text_color` in the `fig.add_annotation` call?
Let's check the code of `_create_empty_state_figure` again.
