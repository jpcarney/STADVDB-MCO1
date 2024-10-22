import kagglehub
from dash import Dash, html, dash_table
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# Download latest version
path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

print("Path to dataset files:", path)

# extract
data = pd.read_csv("C:/Users/narut/.cache/kagglehub/datasets/fronkongames/steam-games-dataset/versions/29/games.csv")
df = pd.DataFrame(data)

# reset the index and move the index values to a new 'AppID' column
df.reset_index(inplace=True)

# shift values from columns 'AppID' to 'DiscountDLC count' by 1 position to the right
cols_to_shift = df.columns[1:df.columns.get_loc('DiscountDLC count') + 1]

# create new df with shifted columns
df[cols_to_shift] = df[cols_to_shift].shift(periods=1, axis=1)

# move index values to the 'AppID' column
df['AppID'] = df['index']

# drop temp 'index' column
df.drop(columns=['index'], inplace=True)
print(data.columns)

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