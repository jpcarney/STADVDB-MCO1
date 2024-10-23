import kagglehub
from dash import Dash, html, dash_table
import pandas as pd
import numpy as np
import pymysql

# Download latest version
path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

print("Path to dataset files:", path)

# Load CSV file
file_path = "C:/Users/Lexrey/.cache/kagglehub/datasets/fronkongames/steam-games-dataset/versions/29/games.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# reset the index and move the index values to a new 'AppID' column
df.reset_index(inplace=True)

# shift values from columns 'AppID' to 'DiscountDLC count' by 1 position to the right
cols_to_shift = df.columns[1:df.columns.get_loc('DiscountDLC count') + 1]

# create new df with shifted columns
df[cols_to_shift] = df[cols_to_shift].shift(periods=1, axis=1)

# move index values to the 'AppID' column
df['AppID'] = df['index']

# Drop the temp 'index' column if exists
if 'index' in df.columns:
    df.drop(columns=['index'], inplace=True)

# Drop the 'score_rank' column if it exists
if 'Score rank' in df.columns:
    df.drop('Score rank', axis=1, inplace=True)

# Replace empty strings with NaN and empty lists with None
df = df.map(lambda x: None if isinstance(x, list) and len(x) == 0 else (np.nan if x == '' else x))

# Replace NaN values with 'None' (which MySQL will interpret as NULL)
df = df.where(pd.notnull(df), None)

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

print(df.head(1))

def load_data(connection, df):
    try:
        # String holder for delete Game Table values SQL query
        delete_game_query = "DELETE FROM Games"

        # String holder for insert Game Table values SQL query
        insert_game_query = """
        INSERT INTO Games (id, name, release_date, required_age, price, dlc_count, 
                           about_the_game, reviews, header_image, website, support_url, support_email, 
                           onWindows, onMac, onLinux, metacritic_score, metacritic_url, 
                           achievements, recommendations, notes, 
                           user_score, positive, negative, estimated_owners, 
                           average_playtime_forever, average_playtime_2weeks, 
                           median_playtime_forever, median_playtime_2weeks, peak_ccu)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        delete_movies_query = "DELETE FROM GameMovies"
        insert_movies_query = "INSERT INTO GameMovies (game_id, movies_link) VALUES (%s, %s)"

        delete_screenshots_query = "DELETE FROM GameScreenshots"
        insert_screenshot_query = "INSERT INTO GameScreenshots (game_id, screenshot_link) VALUES (%s, %s)"

        with connection.cursor() as cursor:
            # Delete existing records in the Games table
            cursor.execute(delete_game_query)
            cursor.execute(delete_movies_query)
            cursor.execute(delete_screenshots_query)
            print("All records deleted.")

            # Insert data from the dataframe
            for _, row in df.iterrows():
                try:
                    game_data = (
                        row['AppID'], row['Name'], row['Release date'], row['Required age'], row['Price'],
                        row['DiscountDLC count'], row['About the game'], row['Reviews'],
                        row['Header image'], row['Website'], row['Support url'],
                        row['Support email'], row['Windows'], row['Mac'],
                        row['Linux'], row['Metacritic score'], row['Metacritic url'],
                        row['Achievements'], row['Recommendations'], row['Notes'],
                        row['User score'], row['Positive'], row['Negative'], row['Estimated owners'],
                        row['Average playtime forever'], row['Average playtime two weeks'],
                        row['Median playtime forever'], row['Median playtime two weeks'], row['Peak CCU']
                    )

                    # Execute insert for the Games table
                    cursor.execute(insert_game_query, game_data)

                    # Retrieve the game ID from the current row of the DataFrame using the 'AppID' column
                    game_id = row['AppID']

                    # Insert Screenshots
                    if 'Screenshots' in row and row['Screenshots']:
                        screenshot_links = row['Screenshots'].split(',')  # Split the string into a list of links
                        for screenshot_link in screenshot_links:
                            cursor.execute(insert_screenshot_query,
                                           (game_id, screenshot_link.strip()))  # Insert each link

                    # Insert Movies
                    if 'Movies' in row and row['Movies']:
                        movies_links = row['Movies'].split(',')  # Split the string into a list of links
                        for movies_link in movies_links:
                            cursor.execute(insert_movies_query, (game_id, movies_link.strip()))  # Insert each link

                except Exception as inner_e:
                    print(f"Error inserting row {row['AppID']}: {inner_e}")
                    continue  # Skip this row and continue with the next

            connection.commit()
            print("Data loaded successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123',
    'database': 'steam'
}

# Create a connection using pymysql
try:
    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
		autocommit=True
    )

    # Load the data into the database
    load_data(connection, df)

except Exception as e:
    print(f"Error connecting to the database: {e}")

finally:
    # Close the connection
    if connection:
        connection.close()

if __name__ == '__main__':
    app.run(debug=True)