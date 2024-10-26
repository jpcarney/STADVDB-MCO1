import ast
import kagglehub
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px
import os
from sqlalchemy import create_engine, inspect, text
import time
from sqlalchemy.orm import Session
from setuptools.installer import fetch_build_egg

# Download latest version
path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

print("Path to dataset files:", path)

# Load CSV file
file_path = os.path.expanduser("~/.cache/kagglehub/datasets/fronkongames/steam-games-dataset/versions/29/games.csv")
df = pd.read_csv(file_path, encoding='utf-8', nrows=1000)

# reset the index and move the index values to a new 'AppID' column
df.reset_index(inplace=True)

# shift values from columns 'AppID' to 'DiscountDLC count' by 1 position to the right
cols_to_shift = df.columns[1:df.columns.get_loc('DiscountDLC count') + 1]

# create new df with shifted columns
df[cols_to_shift] = df[cols_to_shift].shift(periods=1, axis=1)

# move index values to the 'AppID' column
df['AppID'] = df['index']

# Drop temporary column
df.drop(columns=['index'], inplace=True)

# Drop unneeded columns with more than 50% null values
df.drop(['Notes', 'Score rank', 'Metacritic url', 'Support url', 'Website'], axis=1, inplace=True)

# Drop rows with empty rows
df.dropna(subset=['Name'], inplace=True)

# Replace empty strings with NaN and empty lists with None
df = df.map(lambda x: None if isinstance(x, list) and len(x) == 0 else (np.nan if x == '' else x))

# Replace NaN values with 'None' (which MySQL will interpret as NULL)
df = df.where(pd.notnull(df), None)

# Define a function to parse dates
def parse_dates(date_str):
    # Check if the format is '%b %Y'
    try:
        # Try to parse as '%b %Y'
        return pd.to_datetime(date_str, format='%b %Y').date()
    except ValueError:
        # If it fails, try other formats
        try:
            return pd.to_datetime(date_str, errors='coerce').date()
        except ValueError:
            return None  # Return None if all formats fail

# Apply the function to the 'Release date' column
df['Release date'] = df['Release date'].apply(parse_dates)


def load_data(engine, df):
    try:
        # Insert queries
        insert_game_query = """
            INSERT INTO Games (id, name, release_date, required_age, price, dlc_count, about_the_game, reviews, 
                               header_image, support_email, onWindows, onMac, onLinux, 
                               metacritic_score, achievements, recommendations, user_score, 
                               positive, negative, estimated_owners, average_playtime_forever, average_playtime_2weeks, 
                               median_playtime_forever, median_playtime_2weeks, peak_ccu)
            VALUES (:id, :name, :release_date, :required_age, :price, :dlc_count, :about_the_game, :reviews, 
                    :header_image, :support_email, :onWindows, :onMac, :onLinux, 
                    :metacritic_score, :achievements, :recommendations, :user_score, 
                    :positive, :negative, :estimated_owners, :average_playtime_forever, :average_playtime_2weeks, 
                    :median_playtime_forever, :median_playtime_2weeks, :peak_ccu)
        """
        insert_movies_query = "INSERT INTO GameMovies (game_id, movies_link) VALUES (:game_id, :movies_link)"
        insert_screenshot_query = "INSERT INTO GameScreenshots (game_id, screenshot_link) VALUES (:game_id, :screenshot_link)"
        insert_tag_query = "INSERT IGNORE INTO Tags (tag_name) VALUES (:tag_name)"
        insert_genre_query = "INSERT IGNORE INTO Genres (genre_name) VALUES (:genre_name)"
        insert_category_query = "INSERT IGNORE INTO Categories (category_name) VALUES (:category_name)"
        insert_publisher_query = "INSERT IGNORE INTO Publishers (publisher_name) VALUES (:publisher_name)"
        insert_developer_query = "INSERT IGNORE INTO Developers (developer_name) VALUES (:developer_name)"
        insert_languages_query = "INSERT IGNORE INTO Languages (language_name) VALUES (:language_name)"

        # Linking queries
        insert_game_tag_query = "INSERT IGNORE INTO GameTags (game_id, tag_id) VALUES (:game_id, :tag_id)"
        insert_game_genre_query = "INSERT IGNORE INTO GameGenres (game_id, genre_id) VALUES (:game_id, :genre_id)"
        insert_game_category_query = "INSERT IGNORE INTO GameCategories (game_id, category_id) VALUES (:game_id, :category_id)"
        insert_game_publisher_query = "INSERT IGNORE INTO GamePublishers (game_id, publisher_id) VALUES (:game_id, :publisher_id)"
        insert_game_developer_query = "INSERT IGNORE INTO GameDevelopers (game_id, developer_id) VALUES (:game_id, :developer_id)"
        insert_supported_languages_query = "INSERT IGNORE INTO Supported_Languages (game_id, language_id) VALUES (:game_id, :language_id)"
        insert_full_audio_languages_query = "INSERT IGNORE INTO Full_Audio_Languages (game_id, language_id) VALUES (:game_id, :language_id)"

        with engine.begin() as connection:  # Automatically handles transaction commit/rollback

            # TRUNCATE all tables
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            for table in tables:
                connection.execute(text(f"TRUNCATE TABLE {table};"))
                print(f"Truncated table: {table}")
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            print("All tables truncated successfully.")

            # Prepare batch insert data
            games_data = []
            movies_data = []
            screenshots_data = []
            genres_data = set()
            categories_data = set()
            languages_data = set()
            publishers_data = set()
            developers_data = set()
            tags_data = set()
            game_genres_data = []
            game_categories_data = []
            game_publishers_data = []
            game_developers_data = []
            game_tags_data = []
            supported_languages_data = []
            full_audio_languages_data = []

            for _, row in df.iterrows():
                game_id = row['AppID']
                try:
                    # Games data
                    game_data = {
                        'id': row['AppID'],
                        'name': row['Name'],
                        'release_date': row['Release date'],
                        'required_age': row['Required age'],
                        'price': row['Price'],
                        'dlc_count': row['DiscountDLC count'],
                        'about_the_game': row['About the game'],
                        'reviews': row['Reviews'],
                        'header_image': row['Header image'],
                        'support_email': row['Support email'],
                        'onWindows': row['Windows'],
                        'onMac': row['Mac'],
                        'onLinux': row['Linux'],
                        'metacritic_score': row['Metacritic score'],
                        'achievements': row['Achievements'],
                        'recommendations': row['Recommendations'],
                        'user_score': row['User score'],
                        'positive': row['Positive'],
                        'negative': row['Negative'],
                        'estimated_owners': row['Estimated owners'],
                        'average_playtime_forever': row['Average playtime forever'],
                        'average_playtime_2weeks': row['Average playtime two weeks'],
                        'median_playtime_forever': row['Median playtime forever'],
                        'median_playtime_2weeks': row['Median playtime two weeks'],
                        'peak_ccu': row['Peak CCU']
                    }
                    games_data.append(game_data)
                    # Movies
                    if 'Movies' in row and row['Movies']:
                        movies_links = row['Movies'].split(',')
                        for movies_link in movies_links:
                            movies_data.append({'game_id': game_id, 'movies_link': movies_link.strip()})

                    # Screenshots
                    if 'Screenshots' in row and row['Screenshots']:
                        screenshot_links = row['Screenshots'].split(',')
                        for screenshot_link in screenshot_links:
                            screenshots_data.append({'game_id': game_id, 'screenshot_link': screenshot_link.strip()})

                    # Genres
                    if 'Genres' in row and row['Genres']:
                        genre_list = row['Genres'].split(',')
                        for genre_name in genre_list:
                            genres_data.add(genre_name.strip())
                            game_genres_data.append((game_id, genre_name.strip()))

                    # Categories
                    if 'Categories' in row and row['Categories']:
                        category_list = row['Categories'].split(',')
                        for category_name in category_list:
                            categories_data.add(category_name.strip())
                            game_categories_data.append((game_id, category_name.strip()))

                    # Supported Languages
                    if 'Supported languages' in row and row['Supported languages']:
                        supported_languages_list = ast.literal_eval(row['Supported languages'])
                        for language_name in supported_languages_list:
                            language_name = language_name.strip()
                            languages_data.add(language_name)
                            supported_languages_data.append((game_id, language_name))

                    # Full Audio Languages
                    if 'Full audio languages' in row and row['Full audio languages']:
                        audio_languages_list = ast.literal_eval(row['Full audio languages'])
                        for language_name in audio_languages_list:
                            language_name = language_name.strip()
                            languages_data.add(language_name)
                            full_audio_languages_data.append((game_id, language_name))

                    # Publishers
                    if 'Publishers' in row and row['Publishers']:
                        publisher_list = row['Publishers'].split(',')
                        for publisher_name in publisher_list:
                            publishers_data.add(publisher_name.strip())
                            game_publishers_data.append((game_id, publisher_name.strip()))

                    # Developers
                    if 'Developers' in row and row['Developers']:
                        developer_list = row['Developers'].split(',')
                        for developer_name in developer_list:
                            developers_data.add(developer_name.strip())
                            game_developers_data.append((game_id, developer_name.strip()))

                    # Tags
                    if 'Tags' in row and row['Tags']:
                        tag_list = row['Tags'].split(',')
                        for tag_name in tag_list:
                            tags_data.add(tag_name.strip())
                            game_tags_data.append((game_id, tag_name.strip()))

                except KeyError as ke:
                    print(f"Missing key {ke} in row {game_id}, skipping...")
                    continue
                except Exception as inner_e:
                    print(f"Error processing row {game_id}: {inner_e}")
                    continue

            # Execute batch insertions
            try:
                # Insert main data
                print("Inserting games into games table.")
                connection.execute(text(insert_game_query), games_data)

                connection.execute(text(insert_movies_query), movies_data)
                print("Inserted data into movies table.")
                connection.execute(text(insert_screenshot_query), screenshots_data)
                print("Inserted data into screenshots table.")
                connection.execute(text(insert_genre_query), [{'genre_name': genre} for genre in genres_data])
                print("Inserted data into genres table.")
                connection.execute(text(insert_category_query), [{'category_name': category} for category in categories_data])
                print("Inserted data into categories table.")
                connection.execute(text(insert_languages_query), [{'language_name': language} for language in languages_data])
                print("Inserted data into languages table.")
                connection.execute(text(insert_publisher_query), [{'publisher_name': publisher} for publisher in publishers_data])
                print("Inserted data into publishers table.")
                connection.execute(text(insert_developer_query), [{'developer_name': developer} for developer in developers_data])
                print("Inserted data into developers table.")
                connection.execute(text(insert_tag_query), [{'tag_name': tag} for tag in tags_data])
                print("Inserted data into tags table.")

                print("Inserted data into main tables.")

                # Fetch all IDs in bulk for linking
                result = connection.execute(text("SELECT id, genre_name FROM Genres"))
                genre_id_map = {name: id for id, name in result.fetchall()}

                result = connection.execute(text("SELECT id, category_name FROM Categories"))
                category_id_map = {name: id for id, name in result.fetchall()}

                result = connection.execute(text("SELECT id, language_name FROM Languages"))
                language_id_map = {name: id for id, name in result.fetchall()}

                result = connection.execute(text("SELECT id, publisher_name FROM Publishers"))
                publisher_id_map = {name: id for id, name in result.fetchall()}

                result = connection.execute(text("SELECT id, developer_name FROM Developers"))
                developer_id_map = {name: id for id, name in result.fetchall()}

                result = connection.execute(text("SELECT id, tag_name FROM Tags"))
                tag_id_map = {name: id for id, name in result.fetchall()}

                # Prepare link data for batch insertion
                game_genres_link_data = [
                    {'game_id': game_id, 'genre_id': genre_id_map[genre_name]}
                    for game_id, genre_name in game_genres_data if genre_name in genre_id_map
                ]

                game_categories_link_data = [
                    {'game_id': game_id, 'category_id': category_id_map[category_name]}
                    for game_id, category_name in game_categories_data if category_name in category_id_map
                ]

                game_publishers_link_data = [
                    {'game_id': game_id, 'publisher_id': publisher_id_map[publisher_name]}
                    for game_id, publisher_name in game_publishers_data if publisher_name in publisher_id_map
                ]

                supported_languages_link_data = [
                    {'game_id': game_id, 'language_id': language_id_map[language_name]}
                    for game_id, language_name in supported_languages_data if language_name in language_id_map
                ]

                full_audio_languages_link_data = [
                    {'game_id': game_id, 'language_id': language_id_map[language_name]}
                    for game_id, language_name in full_audio_languages_data if language_name in language_id_map
                ]

                game_developers_link_data = [
                    {'game_id': game_id, 'developer_id': developer_id_map[developer_name]}
                    for game_id, developer_name in game_developers_data if developer_name in developer_id_map
                ]

                game_tags_link_data = [
                    {'game_id': game_id, 'tag_id': tag_id_map[tag_name]}
                    for game_id, tag_name in game_tags_data if tag_name in tag_id_map
                ]

                # Linking
                connection.execute(text(insert_game_tag_query), game_tags_link_data)
                print("Inserted data into GameTags table.")
                connection.execute(text(insert_game_genre_query), game_genres_link_data)
                print("Inserted data into GameGenres table.")
                connection.execute(text(insert_game_category_query), game_categories_link_data)
                print("Inserted data into GameCategories table.")
                connection.execute(text(insert_game_publisher_query), game_publishers_link_data)
                print("Inserted data into GamePublishers table.")
                connection.execute(text(insert_game_developer_query), game_developers_link_data)
                print("Inserted data into GameDevelopers table.")
                connection.execute(text(insert_supported_languages_query), supported_languages_link_data)
                print("Inserted data into Supported_Languages table.")
                connection.execute(text(insert_full_audio_languages_query), full_audio_languages_link_data)
                print("Inserted data into Full_Audio_Languages table.")

                print("Data loaded successfully.")
            except Exception as e:
                print(f"Error while inserting data: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")


def fetch_roll_up_data(engine, selected_column, grouping):
    if selected_column == "Release date":
        if grouping == "Year":
            query = """
            SELECT YEAR(release_date) AS release_year, COUNT(id) AS num_games
            FROM Games
            GROUP BY release_year;
            """
        elif grouping == "Month":
            query = """
            SELECT DATE_FORMAT(release_date, '%Y-%m') AS release_month, COUNT(id) AS num_games
            FROM Games
            GROUP BY release_month;
            """
    elif selected_column == "Genre":
        query = """
        SELECT Genres.genre_name, COUNT(Games.id) AS num_games
        FROM Games
        JOIN GameGenres ON Games.id = GameGenres.game_id
        JOIN Genres ON GameGenres.genre_id = Genres.id
        GROUP BY Genres.genre_name;
        """

    with engine.connect() as connection:
        return pd.read_sql(query, connection)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123',
    'database': 'steam',
    'dialect': 'mysql+pymysql'  # Specify the dialect and driver
}

# Create a connection using SQLAlchemy
try:
    # Create the SQLAlchemy engine
    engine = create_engine(f"{db_config['dialect']}://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}", echo=True)

    # Load the data into the database
    load_data(engine, df)  # Modify load_data to accept an SQLAlchemy engine

except Exception as e:
    print(f"Error connecting to the database: {e}")

# https://github.com/cordb/gutensearch

# Initialize the Dash app with Bootstrap styling
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Roll-up', value='tab-1'),
        dcc.Tab(label='Drill-down', value='tab-2'),
        dcc.Tab(label='Slice', value='tab-3'),
        dcc.Tab(label='Dice', value='tab-4'),
        dcc.Tab(label='About', value='tab-5'),
    ], className="tabs-container"),
    html.Div(id='tabs-content', className='dashboard-container')
])

options = [
    {'label': 'Release Date', 'value': 'Release date'},
    {'label': 'Genre', 'value': 'Genre'},
    {'label': 'Price', 'value': 'Price'},
    {'label': 'User Score', 'value': 'User score'},
    {'label': 'Estimated Owners', 'value': 'Estimated owners'},
    {'label': 'Peak CCU', 'value': 'Peak CCU'}
]

# Callback to update the content based on selected tab
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
)

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            # Container for the main dropdown and grouping dropdown
            html.Div([  # Dropdown container
                html.H4("Variable Select:"),
                dcc.Dropdown(
                    id='rollup-dropdown',
                    options=options,
                    value='Name',
                    clearable=False
                ),
                # Grouping Dropdown
                html.Div([
                    html.H4("Group By:"),
                    dcc.Dropdown(
                        id='grouping-selector',
                        options=[
                            {'label': 'Year', 'value': 'Year'},
                            {'label': 'Month', 'value': 'Month'}
                        ],
                        value='Year',  # Default selection
                        clearable=False
                    )
                ], id='grouping-dropdown-container'),  # No longer hidden; now directly below the main dropdown
            ], className='dropdown-container'),  # Class for dropdown styling

            html.Div(id='output-div', className='output-container')  # Output div
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([  # Dropdown container
                html.H4("Variable Select:"),
                dcc.Dropdown(
                    id='drilldown-dropdown',
                    options=options,
                    value='Name',
                    clearable=False
                )
            ], className='dropdown-container'),  # Class for dropdown styling
            html.Div(id='output-div', className='output-container')  # Output div
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div([  # Dropdown container
                html.H4("Variable Select:"),
                dcc.Dropdown(
                    id='slice-dropdown',
                    options=options,
                    value='Name',
                    clearable=False
                )
            ], className='dropdown-container'),  # Class for dropdown styling
            html.Div(id='output-div', className='output-container')  # Output div
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.Div([  # Dropdown container
                html.H4("Variable Select:"),
                dcc.Dropdown(
                    id='dice-dropdown',
                    options=options,
                    value='Name',
                    clearable=False
                )
            ], className='dropdown-container'),  # Class for dropdown styling
            html.Div(id='output-div', className='output-container')  # Output div
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H1("About This Dashboard"),
            html.P("Information about the OLAP operations dashboard goes here.")
        ])
    return html.Div()

# Callback to show/hide the grouping dropdown
@app.callback(
    Output('output-div', 'children'),
    Input('rollup-dropdown', 'value'),
    Input('grouping-selector', 'value'),  # Add this input
    Input('tabs', 'value')  # Include the current tab as input
)

def update_output(selected_value, grouping, current_tab):
    # Only update if we are in tab-1
    if current_tab == 'tab-1':
        # Define a mapping of selected values to DataFrame columns and titles
        options = {
            'Release date': {
                'x': 'release_year',  # Change this to 'release_year' or 'release_month' based on grouping
                'y': 'num_games',
                'title': 'Number of Games by Release Date',
                'x_label': 'Release Year',  # Update accordingly
                'y_label': 'Total Games'
            },
            'Genre': {
                'x': 'genre_name',
                'y': 'num_games',
                'title': 'Number of Games per Genre',
                'x_label': 'Game Genre',
                'y_label': 'Total Games'
            },
            'Price': {
                'x': 'price',
                'y': 'num_games',
                'title': 'Number of Games by Price',
                'x_label': 'Price',
                'y_label': 'Total Games'
            },
            'User score': {
                'x': 'user_score',
                'y': 'num_games',
                'title': 'Number of Games by User Score',
                'x_label': 'User Score',
                'y_label': 'Total Games'
            },
            'Estimated owners': {
                'x': 'estimated_owners',
                'y': 'num_games',
                'title': 'Number of Games by Estimated Owners',
                'x_label': 'Estimated Owners',
                'y_label': 'Total Games'
            },
            'Peak CCU': {
                'x': 'peak_ccu',
                'y': 'num_games',
                'title': 'Number of Games by Peak CCU',
                'x_label': 'Peak CCU',
                'y_label': 'Total Games'
            }
        }

        df = fetch_roll_up_data(engine, selected_value, grouping)
        if selected_value in options:
            params = options[selected_value]

            # Adjust x column based on grouping if needed
            if selected_value == 'Release date':
                params['x'] = 'release_year' if grouping == 'Year' else 'release_month'

            df = df.sort_values(by=params['y'], ascending=False)
            fig = px.bar(df, x=params['x'], y=params['y'],
                         title=params['title'],
                         labels={params['x']: params['x_label'], params['y']: params['y_label']})
            return dcc.Graph(figure=fig)

    return html.Div()  # Return empty div for other tabs
@app.callback(
    Output('grouping-dropdown-container', 'style'),
    Input('rollup-dropdown', 'value')
)
def update_grouping_dropdown(selected_value):
    if selected_value == 'Release date':
        return {'display': 'block'}  # Show the dropdown
    return {'display': 'none'}  # Hide the dropdown

if __name__ == '__main__':
    app.run_server(debug=False)