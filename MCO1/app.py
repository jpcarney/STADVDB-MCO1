import kagglehub
from dash import Dash, html, dash_table
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# Download latest version
path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

print("Path to dataset files:", path)

# extract
data = pd.read_csv("C:/Users/narut/.cache/kagglehub/datasets/fronkongames/steam-games-dataset/versions/29/games.csv", index_col=False)
df = pd.DataFrame(data)

# Transform / cleaning
# Identify the starting index for the shift operation
start_column = "DiscountDLC count"
start_index = df.columns.get_loc(start_column)

# generalize all column as object to prevent forcing NaN due to incompatible dtypes
df[df.columns] = df[df.columns].astype('object')

for i in range(start_index + 1, len(df.columns)):
    df.iloc[:, i - 1] = df.iloc[:, i]  # Move each value to the left


# Initialize the Dash app
app = Dash(__name__)

# Create the layout with the DataTable to display the DataFrame
app.layout = html.Div([
    html.H1("Steam Games Dataset"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],  # Create column names from the DataFrame
        data=df.head(5).to_dict('records'),  # Convert DataFrame to a list of dictionaries
        page_size=1,  # Set the number of rows per page
        style_table={'overflowX': 'auto'},  # Allow horizontal scrolling if necessary
        style_cell={
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': 'lightgrey',
            'color': 'black'
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)