with open("ivi_water/cli.py", "r") as f:
    content = f.read()

idx = content.find("def get_spatial_units")
print(content[idx:idx+1500])
