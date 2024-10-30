import ast
import kagglehub
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px
import os
from sqlalchemy import create_engine, inspect, text
import pymysql
from setuptools.installer import fetch_build_egg
import time

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

# Replace "0 - 0" with "0 - 20000" in the Estimated Owners column
df['Estimated owners'] = df['Estimated owners'].replace("0 - 0", "0 - 20000")

# function to parse dates
def parse_dates(date_str):
    # Check if the format is '%b %Y'
    try:
        # Try to parse as '%b %Y'
        return pd.to_datetime(date_str, format='%b %Y').date()
    except ValueError:
        try:
            return pd.to_datetime(date_str, errors='coerce').date()
        except ValueError:
            return None

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


def fetch_roll_up_drill_down_data(engine, selected_column, grouping):
    if selected_column == "Release date":
        if grouping == "5 Years":
            query = """
            SELECT CONCAT(FLOOR(YEAR(release_date) / 5) * 5, '-', FLOOR(YEAR(release_date) / 5) * 5 + 4) 
            AS release_5years,
                   COUNT(id) AS num_games
            FROM Games
            GROUP BY release_5years;
            """
        if grouping == "Year":
            query = """
            SELECT YEAR(release_date) AS release_year, COUNT(id) AS num_games
            FROM Games
            GROUP BY release_year;
            """
        elif grouping == "Month":
            query = """
            SELECT DATE_FORMAT(release_date, '%%Y-%%m') AS release_month, COUNT(id) AS num_games
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
    elif selected_column == "Price":
        query = """
        SELECT CONCAT(FLOOR(price / 5) * 5, '-', FLOOR(price / 5) * 5 + 4.99) AS price_range,
               COUNT(id) AS num_games
        FROM Games
        GROUP BY price_range
        """
    elif selected_column == "Estimated owners":
        query = """
        SELECT estimated_owners, COUNT(id) AS num_games
        FROM Games
        GROUP BY estimated_owners
        """
    elif selected_column == "Peak CCU":
        query = """
        SELECT 
            CASE 
                WHEN peak_ccu < 100 THEN '0-99'
                WHEN peak_ccu < 200 THEN '100-199'
                WHEN peak_ccu < 500 THEN '200-499'
                WHEN peak_ccu < 1000 THEN '500-999'
                ELSE '1000+' 
            END AS ccu_range,
            COUNT(id) AS num_games
        FROM Games
        GROUP BY ccu_range
        ORDER BY MIN(peak_ccu);
        """
    with engine.connect() as connection:
        return pd.read_sql(query, connection)

def fetch_slice_data(engine, selected_column, grouping):
    if selected_column == "Genres" and grouping:
        query = """
        SELECT Genres.genre_name, estimated_owners, COUNT(Genres.genre_name) AS owned_in_genre
        FROM Games
        JOIN GameGenres ON Games.id = GameGenres.game_id
        JOIN Genres ON GameGenres.genre_id = Genres.id
        WHERE Genres.genre_name = %s
        GROUP BY Genres.genre_name, estimated_owners
        ORDER BY estimated_owners ASC, owned_in_genre ASC;
        """

        with engine.connect() as connection:
            return pd.read_sql(query, connection, params=(grouping,))
    elif selected_column == "Categories" and grouping:
        query = """
        SELECT Categories.category_name, estimated_owners, COUNT(Categories.category_name) AS owned_in_category
        FROM Games
        JOIN GameCategories ON Games.id = GameCategories.game_id
        JOIN Categories ON GameCategories.category_id = Categories.id
        WHERE Categories.category_name = %s
        GROUP BY Categories.category_name, estimated_owners
        ORDER BY estimated_owners ASC, owned_in_category ASC;
        """

        with engine.connect() as connection:
            return pd.read_sql(query, connection, params=(grouping,))

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

    start_time = time.time()

    # Load the data into the database
    load_data(engine, df)  # Modify load_data to accept an SQLAlchemy engine

    end_time = time.time()

    # Calculate the runtime
    runtime = end_time - start_time
    print(f"Runtime: {runtime} seconds")

except Exception as e:
    print(f"Error connecting to the database: {e}")

# https://github.com/cordb/gutensearch

# Initialize the Dash app with Bootstrap styling
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Roll-up/Drill-down', value='tab-1'),
        dcc.Tab(label='Slice', value='tab-2'),
        dcc.Tab(label='Dice', value='tab-3'),
        dcc.Tab(label='About', value='tab-4'),
    ], className="tabs-container"),
    html.Div(id='tabs-content', className='dashboard-container')
])

options = [
    {'label': 'Release Date', 'value': 'Release date'},
    {'label': 'Genre', 'value': 'Genre'},
    {'label': 'Price', 'value': 'Price'},
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
        return create_rollup_drilldown_content()
    elif tab == 'tab-2':
        return create_slice_content()
    elif tab == 'tab-3':
        return create_dice_content()
    elif tab == 'tab-4':
        return html.Div([
            html.H1("About This Dashboard"),
            html.P(
                "This dashboard is an OLAP Web Application designed for visualizing and analyzing data outputs from the steam games database as a requirement for STADVDB MCO1."),
            html.P("Project Details:"),
            html.Ul([
                html.Li([html.Strong("Project Name: "), "STADVDB-MCO1"]),
                html.Li([html.Strong("Version: "), "1.0"]),
                html.Li([html.Strong("Developers: "), "John Paul C. Carney, Lexrey D. Porciuncula"]),
                html.Li([html.Strong("Contact Information: "), "jpcarney@dlsu.edu.ph, lexrey_porciuncula@dlsu.edu.ph"]),
                html.Li([html.Strong("Description: "), "OLAP Web Application for STADVDB MCO1 Output"]),
                html.Li([
                    html.Strong("Repository: "),
                    html.A("GitHub Repository", href="https://github.com/jpcarney/STADVDB-MCO1")
                ]),
                html.Li([html.Strong("Required Packages: "),
                         "kagglehub, dash, pandas, numpy, dash_bootstrap_components, plotly, sqlalchemy"]),
            ]),
        ])

def create_rollup_drilldown_content():
    return html.Div([
        html.Div([  # Dropdown container
            html.H4("Variable Select:"),
            dcc.Dropdown(
                id='rollup-drilldown-dropdown',
                options=options,
                value='Release date',
                clearable=False
            ),
            html.Div(id='grouping-dropdown-container', children=[
                html.H4("Group By:"),
                dcc.Dropdown(
                    id='grouping-selector',
                    options=[
                        {'label': '5 Years', 'value': '5 Years'},
                        {'label': 'Year', 'value': 'Year'},
                        {'label': 'Month', 'value': 'Month'}
                    ],
                    value='Year',
                    clearable=False
                )
            ]),
        ], className='dropdown-container'),
        html.Div(id='rollup-drilldown-output', className='output-container')  # Output div
    ])

# List of genres
genres = sorted([
    'Early Access', 'Audio Production', 'Sports', 'Racing',
    'Web Publishing', 'Violent', 'Gore', 'Sexual Content',
    'Adventure', 'RPG', 'Photo Editing', 'Utilities',
    'Action', 'Game Development', 'Video Production',
    'Massively Multiplayer', 'Design & Illustration', 'Indie',
    'Nudity', 'Software Training', 'Education', 'Simulation',
    'Free to Play', 'Animation & Modeling', 'Casual', 'Strategy'
])

categories = sorted([
    'Remote Play on TV', 'Steam Leaderboards', 'MMO', 'In-App Purchases',
    'Shared/Split Screen PvP', 'Steam Trading Cards', 'Single-player', 'Co-op',
    'Steam Workshop', 'VR Support', 'SteamVR Collectibles', 'Partial Controller Support',
    'Remote Play on Phone', 'Captions available', 'Shared/Split Screen Co-op',
    'LAN PvP', 'Stats', 'PvP', 'Multi-player', 'Steam Cloud',
    'Remote Play Together', 'LAN Co-op', 'Steam Achievements', 'Online PvP',
    'Shared/Split Screen', 'Full controller support', 'Includes level editor',
    'Cross-Platform Multiplayer', 'Online Co-op', 'Remote Play on Tablet',
    'Steam Turn Notifications'
])

def create_slice_content():
    return html.Div([
        html.Div([  # Dropdown container for variable selection
            html.H4("Variable Select:"),
            dcc.Dropdown(
                id='slice-variable-dropdown',
                options=[
                    {'label': 'Genres', 'value': 'Genres'},
                    {'label': 'Categories', 'value': 'Categories'},
                ],
                value='Genres',  # Default selection
                clearable=False
            ),
            # Genre Grouping Dropdown
            html.Div(id='genre-grouping-dropdown-container', children=[
                html.H4("Group By Genre:"),
                dcc.Dropdown(
                    id='genre-slice-selector',
                    options=[{'label': genre, 'value': genre} for genre in genres],
                    value='Early Access',  # Default selection
                    clearable=False
                ),
                html.Div(id='slice-output', className='output-container')  # Output div for genre
            ]),
            # Categories Grouping Dropdown
            html.Div(id='categories-grouping-dropdown-container', children=[
                html.H4("Group By Category:"),
                dcc.Dropdown(
                    id='category-slice-selector',
                    options=[{'label': category, 'value': category} for category in categories],
                    value='Captions available',  # Default selection
                    clearable=False
                )
            ]),
        ], className='dropdown-container'),
        html.Div(id='slice-output', className='output-container')  # Output div
    ])

# Callback to update the visibility of the dropdowns
@app.callback(
    [
        Output('genre-grouping-dropdown-container', 'style'),
        Output('categories-grouping-dropdown-container', 'style'),
    ],
    Input('slice-variable-dropdown', 'value')
)
def update_grouping_dropdowns(selected_variable):
    genre_style = {'display': 'block'} if selected_variable == 'Genres' else {'display': 'none'}
    categories_style = {'display': 'block'} if selected_variable == 'Categories' else {'display': 'none'}

    return genre_style, categories_style

def create_dice_content():
    return html.Div([
        html.Div([  # Dropdown container
            html.H4("Variable Select:"),
            dcc.Dropdown(
                id='dice-dropdown',
                options=options,
                value='Release date',
                clearable=False
            )
        ], className='dropdown-container'),
        html.Div(id='dice-output', className='output-container')  # Output div
    ])

# Callback for Roll-up/Drill-down tab
@app.callback(
    Output('rollup-drilldown-output', 'children'),
    Input('rollup-drilldown-dropdown', 'value'),
    Input('grouping-selector', 'value'),
)
def update_rollup_drilldown_output(selected_value, grouping):
    return create_output_graph(selected_value, grouping, 'Roll-up/Drill-down')

# Callback for showing/hiding the Group By dropdown
@app.callback(
    Output('grouping-dropdown-container', 'style'),
    Input('rollup-drilldown-dropdown', 'value'),
)
def toggle_grouping_dropdown(selected_value):
    if selected_value == 'Release date':
        return {'display': 'block'}  # Show Group By dropdown
    return {'display': 'none'}  # Hide Group By dropdown

# Callback for Slice tab
@app.callback(
    Output('slice-output', 'children'),  # Update genre output
    Input('slice-variable-dropdown', 'value'),
    Input('genre-slice-selector', 'value'),
    Input('category-slice-selector', 'value')
)
def update_slice_output(selected_value, genre_grouping, category_grouping):
    # Determine the grouping based on the selected variable
    if selected_value == 'Genres':
        grouping = genre_grouping
        return create_output_graph(selected_value, grouping, 'Slice')
    else:
        grouping = category_grouping
        return create_output_graph(selected_value, grouping, 'Slice')

# Callback for Dice tab
@app.callback(
    Output('dice-output', 'children'),
    Input('dice-dropdown', 'value'),
)
def update_dice_output(selected_value):
    return create_output_graph(selected_value, None, 'Dice')

# General function to create output graphs
def create_output_graph(selected_value, grouping, operation):
    options = {
        'Release date': {'x': 'release_year', 'y': 'num_games', 'title': 'Number of Games by Release Date'},
        'Genre': {'x': 'genre_name', 'y': 'num_games', 'title': 'Number of Games per Genre'},
        'Price': {'x': 'price_range', 'y': 'num_games', 'title': 'Number of Games by Price'},
        'Estimated owners': {'x': 'estimated_owners', 'y': 'num_games', 'title': 'Number of Games by Estimated Owners'},
        'Peak CCU': {'x': 'ccu_range', 'y': 'num_games', 'title': 'Number of Games by Peak CCU'},
    }

    slice_options = {
        'Genres': {'x': 'estimated_owners', 'y': 'num_games', 'title': 'Number of Estimated Owners for Genre: ' + grouping},
        'Categories': {'x': 'estimated_owners', 'y': 'num_games', 'title': 'Number of Estimated Owners for Categories: ' + grouping}
    }

    if operation == 'Roll-up/Drill-down':
        df = fetch_roll_up_drill_down_data(engine, selected_value, grouping)
    elif operation == 'Slice':
        df = fetch_slice_data(engine, selected_value, grouping)

    if selected_value in options:
        params = options[selected_value]

        if operation == 'Roll-up/Drill-down':
            if selected_value == 'Release date':
                if grouping == 'Month':
                    params['x'] = 'release_month'
                    params['x_label'] = 'Release Month'
                    df = df.sort_values(by=params['y'], ascending=False)
                elif grouping == 'Year':
                    params['x'] = 'release_year'
                    params['x_label'] = 'Release Year'
                    df = df.sort_values(by=params['y'], ascending=False)
                elif grouping == '5 Years':
                    params['x'] = 'release_5years'
                    params['x_label'] = 'Release Period'
                    df = df.sort_values(by=params['x'], ascending=True)
            elif selected_value == 'Genre':
                df = df.sort_values(by=params['y'], ascending=False)
            elif selected_value == 'Price':
                df['lower_bound'] = df['price_range'].apply(lambda x: float(x.split('-')[0]))
                df = df.sort_values(by='lower_bound', ascending=True)
            elif selected_value == 'Estimated owners':
                df['lower_bound'] = df['estimated_owners'].apply(lambda x: float(x.split('-')[0]) if '-' in x else float(x))
                df = df.sort_values(by='lower_bound', ascending=True)
            elif selected_value == 'Peak CCU':
                df['lower_bound'] = df['ccu_range'].apply(
                    lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', ''))
                )
                df = df.sort_values(by='lower_bound', ascending=True)

        if df is not None:
            fig = px.bar(df, x=params['x'], y=params['y'],
                         title=params['title'],
                         labels={params['x']: params['x_label'], params['y']: 'Total Games'})
            return dcc.Graph(figure=fig)

    if selected_value in slice_options:
        params = slice_options[selected_value]
        if operation == 'Slice':
                # Fetching data using the slice fetch function
                if selected_value in ('Genres', 'Categories'):
                    # Since the grouping is already passed as a parameter
                    df = fetch_slice_data(engine, selected_value, grouping)

                    if df is not None and not df.empty:
                        # Set parameters for the graphs based on fetched data
                        params['x'] = 'estimated_owners'
                        params['y'] = 'owned_in_genre' if selected_value == 'Genres' else 'owned_in_category'
                        params['x_label'] = 'Estimated Owners'
                        params['title'] = slice_options[selected_value]['title']

                        # Create the figure
                        fig = px.bar(df, x=params['x'], y=params['y'],
                                     title=params['title'],
                                     labels={params['x']: params['x_label'], params['y']: 'Number of Games'})
                        return dcc.Graph(figure=fig)

    return html.Div("No data available for the selected variable.")

if __name__ == '__main__':
    app.run_server(debug=False)