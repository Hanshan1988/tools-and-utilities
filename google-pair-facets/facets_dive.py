# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python36
# ---

# +
# Load UCI census and convert to json for sending to the visualization
import pandas as pd
features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"]

# Load dataframe from external CSV and add header information
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    names=features, # name features for header row
    sep=r'\s*,\s*', # separator used in this dataset
    engine='python',
    skiprows=[0], # skip first row without data 
    na_values="?") # add ? where data is missing

# set the sprite_size based on the number of records in dataset,
# larger datasets can crash the browser if the size is too large (>50000)
sprite_size = 32 if len(df.index)>50000 else 64

jsonstr = df.to_json(orient='records')

# +
# Display the Dive visualization for this data
from IPython.core.display import display, HTML

# Create Facets template  
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Title</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/0.7.24/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="900"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>
</head>
<body></body>
</html>
"""

# Load the json dataset and the sprite_size into the template
html = HTML_TEMPLATE.format(jsonstr=jsonstr, sprite_size=sprite_size)

f = open("dive_demo.html","w")
f.write(html)
f.close()
# -


