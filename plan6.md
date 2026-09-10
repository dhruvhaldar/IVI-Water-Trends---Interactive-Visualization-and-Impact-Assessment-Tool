Let's see if the text_color is used inside `_create_empty_state_figure`.
```python
python -c "
with open('ivi_water/visualizer.py', 'r') as f:
    code = f.read()
start = code.find('def _create_empty_state_figure')
end = code.find('def create_seasonal_stacked_area_chart')
print(code[start:end])
"
```
