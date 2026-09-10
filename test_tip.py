import re

with open("ivi_water/cli.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "💡 Tip: Use these location IDs with" in line:
        print(f"Line {i+1}: {line.strip()}")
    elif "💡 Tip: Visualize these trends with" in line:
        print(f"Line {i+1}: {line.strip()}")
    elif "💡 Tip: Create a comprehensive dashboard with" in line:
        print(f"Line {i+1}: {line.strip()}")
    elif "💡 Tip: Generate a full report with" in line:
        print(f"Line {i+1}: {line.strip()}")
    elif "💡 Tip: Open the generated report file in your browser to view the insights." in line:
        print(f"Line {i+1}: {line.strip()}")
    elif "💡 Tip: Start Jupyter notebook to begin your analysis." in line:
        print(f"Line {i+1}: {line.strip()}")
