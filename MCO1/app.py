import ast

import kagglehub
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import pymysql
import dash_bootstrap_components as dbc
import plotly.express as px
from setuptools.installer import fetch_build_egg

# Download latest version
path = kagglehub.dataset_download("fronkongames/steam-games-dataset")

print("Path to dataset files:", path)

# Load CSV file
file_path = "C:/Users/Lexrey/.cache/kagglehub/datasets/fronkongames/steam-games-dataset/versions/29/games.csv"
df = pd.read_csv(file_path, encoding='utf-8', nrows=1000) # nrows = 100 for testing, remove in production

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

def load_data(connection, df):
    try:
        # Delete Queries
        delete_queries = [
            "DELETE FROM Games",
            "DELETE FROM GameMovies",
            "DELETE FROM GameScreenshots",
            "DELETE FROM Tags",
            "DELETE FROM Genres",
            "DELETE FROM Categories",
            "DELETE FROM Publishers",
            "DELETE FROM Developers",
            "DELETE FROM GameTags",
            "DELETE FROM GameGenres",
            "DELETE FROM GameCategories",
            "DELETE FROM GamePublishers",
            "DELETE FROM GameDevelopers",
            "DELETE FROM Languages",
            "DELETE FROM Supported_Languages",
            "DELETE FROM Full_Audio_Languages"
        ]

        # Insert queries
        insert_game_query = """
            INSERT INTO Games (id, name, release_date, required_age, price, dlc_count, about_the_game, reviews, 
                               header_image, support_email, onWindows, onMac, onLinux, 
                               metacritic_score, achievements, recommendations, user_score, 
                               positive, negative, estimated_owners, average_playtime_forever, average_playtime_2weeks, 
                               median_playtime_forever, median_playtime_2weeks, peak_ccu)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s)
        """
        insert_movies_query = "INSERT INTO GameMovies (game_id, movies_link) VALUES (%s, %s)"
        insert_screenshot_query = "INSERT INTO GameScreenshots (game_id, screenshot_link) VALUES (%s, %s)"
        insert_tag_query = "INSERT IGNORE INTO Tags (tag_name) VALUES (%s)"
        insert_genre_query = "INSERT IGNORE INTO Genres (genre_name) VALUES (%s)"
        insert_category_query = "INSERT IGNORE INTO Categories (category_name) VALUES (%s)"
        insert_publisher_query = "INSERT IGNORE INTO Publishers (publisher_name) VALUES (%s)"
        insert_developer_query = "INSERT IGNORE INTO Developers (developer_name) VALUES (%s)"
        insert_languages_query = "INSERT IGNORE INTO Languages (language_name) VALUES (%s)"

        # Linking queries
        insert_game_tag_query = "INSERT INTO GameTags (game_id, tag_id) VALUES (%s, %s)"
        insert_game_genre_query = "INSERT INTO GameGenres (game_id, genre_id) VALUES (%s, %s)"
        insert_game_category_query = "INSERT IGNORE INTO GameCategories (game_id, category_id) VALUES (%s, %s)"
        insert_game_publisher_query = "INSERT IGNORE INTO GamePublishers (game_id, publisher_id) VALUES (%s, %s)"
        insert_game_developer_query = "INSERT IGNORE INTO GameDevelopers (game_id, developer_id) VALUES (%s, %s)"
        insert_supported_languages_query = "INSERT IGNORE INTO Supported_Languages (game_id, language_id) VALUES (%s, %s)"
        insert_full_audio_languages_query = "INSERT IGNORE INTO Full_Audio_Languages (game_id, language_id) VALUES (%s, %s)"

        with connection.cursor() as cursor:
            # Clear tables
            for query in delete_queries:
                cursor.execute(query)
            print("Cleared tables successfully.")

            # Reset AUTO_INCREMENT values to 1
            cursor.execute("ALTER TABLE GameMovies AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE GameScreenshots AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE Tags AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE Genres AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE Categories AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE Publishers AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE Developers AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE Languages AUTO_INCREMENT = 1;")
            print("AUTO_INCREMENT values reset successfully.")

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
                    game_data = (
                        row['AppID'], row['Name'], row['Release date'], row['Required age'], row['Price'],
                        row['DiscountDLC count'], row['About the game'], row['Reviews'],
                        row['Header image'], row['Support email'], row['Windows'], row['Mac'],
                        row['Linux'], row['Metacritic score'], row['Achievements'], row['Recommendations'],
                        row['User score'], row['Positive'], row['Negative'], row['Estimated owners'],
                        row['Average playtime forever'], row['Average playtime two weeks'],
                        row['Median playtime forever'], row['Median playtime two weeks'], row['Peak CCU']
                    )
                    games_data.append(game_data)

                    # Movies
                    if 'Movies' in row and row['Movies']:
                        movies_links = row['Movies'].split(',')
                        for movies_link in movies_links:
                            movies_data.append((game_id, movies_link.strip()))

                    # Screenshots
                    if 'Screenshots' in row and row['Screenshots']:
                        screenshot_links = row['Screenshots'].split(',')
                        for screenshot_link in screenshot_links:
                            screenshots_data.append((game_id, screenshot_link.strip()))

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
                cursor.executemany(insert_game_query, games_data)
                print("Inserted data into games table.")
                cursor.executemany(insert_movies_query, movies_data)
                print("Inserted data into movies table.")
                cursor.executemany(insert_screenshot_query, screenshots_data)
                print("Inserted data into screenshots table.")
                cursor.executemany(insert_genre_query, [(genre,) for genre in genres_data])
                print("Inserted data into genres table.")
                cursor.executemany(insert_category_query, [(category,) for category in categories_data])
                print("Inserted data into categories table.")
                cursor.executemany(insert_languages_query, [(language,) for language in languages_data])
                print("Inserted data into languages table.")

                if publishers_data:
                    cursor.executemany(insert_publisher_query, [(publisher,) for publisher in publishers_data if publisher.strip()])
                    print("Inserted data into publishers table.")
                if developers_data:
                    cursor.executemany(insert_developer_query, [(developer,) for developer in developers_data if developer.strip()])
                cursor.executemany(insert_tag_query, [(tag,) for tag in tags_data])
                print("Inserted data into main tables.")

                # Fetch all IDs in bulk for linking
                cursor.execute("SELECT id, genre_name FROM Genres")
                genre_id_map = {name: id for id, name in cursor.fetchall()}

                cursor.execute("SELECT id, category_name FROM Categories")
                category_id_map = {name: id for id, name in cursor.fetchall()}

                cursor.execute("SELECT id, language_name FROM Languages")
                language_id_map = {name: id for id, name in cursor.fetchall()}

                cursor.execute("SELECT id, publisher_name FROM Publishers")
                publisher_id_map = {name: id for id, name in cursor.fetchall()}

                cursor.execute("SELECT id, developer_name FROM Developers")
                developer_id_map = {name: id for id, name in cursor.fetchall()}

                cursor.execute("SELECT id, tag_name FROM Tags")
                tag_id_map = {name: id for id, name in cursor.fetchall()}

                # Prepare link data for batch insertion
                game_genres_link_data = [(game_id, genre_id_map[genre_name]) for game_id, genre_name in game_genres_data if genre_name in genre_id_map]
                game_categories_link_data = [(game_id, category_id_map[category_name]) for game_id, category_name in game_categories_data if category_name in category_id_map]
                game_publishers_link_data = [(game_id, publisher_id_map[publisher_name]) for game_id, publisher_name in game_publishers_data if publisher_name in publisher_id_map]
                support_languages_link_data = [(game_id, language_id_map[language_name]) for game_id, language_name in supported_languages_data if language_name in language_id_map]
                full_audio_languages_link_data = [(game_id, language_id_map[language_name]) for game_id, language_name in full_audio_languages_data if language_name in language_id_map]
                game_developers_link_data = [(game_id, developer_id_map[developer_name]) for game_id, developer_name in game_developers_data if developer_name in developer_id_map]
                game_tags_link_data = [(game_id, tag_id_map[tag_name]) for game_id, tag_name in game_tags_data if tag_name in tag_id_map]

                # Execute link insertions
                cursor.executemany(insert_game_genre_query, game_genres_link_data)
                cursor.executemany(insert_game_category_query, game_categories_link_data)
                cursor.executemany(insert_supported_languages_query, support_languages_link_data)
                cursor.executemany(insert_full_audio_languages_query, full_audio_languages_link_data)
                cursor.executemany(insert_game_publisher_query, game_publishers_link_data)
                cursor.executemany(insert_game_developer_query, game_developers_link_data)
                cursor.executemany(insert_game_tag_query, game_tags_link_data)
                print("Inserted linked data successfully.")

            except Exception as inner_ex:
                print(f"Error during batch insertion: {inner_ex}")

        connection.commit()
        print("Data loaded successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")


def fetch_roll_up_data(connection):
    query = """
    SELECT Genres.genre_name, COUNT(Games.id) AS num_games
    FROM Games
    JOIN GameGenres ON Games.id = GameGenres.game_id
    JOIN Genres ON GameGenres.genre_id = Genres.id
    GROUP BY Genres.genre_name;
    """
    df = pd.read_sql(query, connection)
    return df

def fetch_drill_down_data(connection):
    query = """
        SELECT Genres.genre_name, Developers.developer_name, COUNT(Games.id) AS num_games
        FROM Games
        JOIN GameGenres ON Games.id = GameGenres.game_id
        JOIN Genres ON GameGenres.genre_id = Genres.id
        JOIN GameDevelopers ON Games.id = GameDevelopers.game_id
        JOIN Developers ON GameDevelopers.developer_id = Developers.id
        GROUP BY Genres.genre_name, Developers.developer_name;
    """
    return pd.read_sql(query, connection)


def update_sunburst(clickData, connection):
    # Fetch data each time the callback is triggered
    data = fetch_drill_down_data(connection)

    # Check if a genre was clicked to drill down
    if clickData:
        selected_genre = clickData["points"][0]["label"]
        filtered_data = data[data["genre_name"] == selected_genre]
    else:
        # Default view: Show all genres with their developers
        filtered_data = data

    # Create the Sunburst chart
    fig = px.sunburst(
        filtered_data,
        path=["genre_name", "dev_name"],  # Define the hierarchy
        values="num_games",
        title="Number of Games by Genre and Developer",
    )

    fig.update_traces(hovertemplate="<b>%{label}</b><br>Games: %{value}<extra></extra>")
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig
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
    {'label': 'Name', 'value': 'Name'},
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
    # ("drilldown-sunburst", "figure"),
    Input('tabs', 'value'),
    # Input("drilldown-sunburst", "clickData")
)

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([  # Return a single Div that contains both components
            html.Div([  # Dropdown container
            html.H4("Variable Select:"),
            dcc.Dropdown(
                id='rollup-dropdown',
                options=options,
                value='Name',
                clearable=False
            )
        ], className='dropdown-container'),  # Class for dropdown styling
            html.Div(id='output-div', className='output-container')  # Output div
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H3("Select Variable:"),
                dcc.Dropdown(
                    id='drilldown-dropdown',
                    options=options,
                    value='Name',
                    clearable=False
                )], className='dropdown-container')
            # dcc.Graph(id="drilldown-sunburst"),
        ])
    elif tab == 'tab-3':
        return html.Div([
                html.Div([
                    html.H3("Select Variable:"),
                    dcc.Dropdown(
                        id='slice-dropdown',
                        options=options,
                        value='Name',
                        clearable=False
                )], className='dropdown-container')
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

@app.callback(
    Output('output-div', 'children'),
    Input('slice-dropdown', 'value'),
    Input('tabs', 'value')  # Include the current tab as input
)
def update_output(selected_value, current_tab):
    # Only update if we are in tab-1
    if current_tab == 'tab-1':
        # Fetch data for roll-up operation
        df = fetch_roll_up_data(connection)
        if selected_value == 'Name':
            return [html.Div("You selected: Name")]
        elif selected_value == 'Genre':
            fig = px.bar(df, x="genre_name", y="num_games", title="Number of Games per Genre")
            return dcc.Graph(figure=fig)
        elif selected_value == 'Release Date':
            return html.Div("You selected: Release Year.")
    return html.Div()  # Return empty div for other tabs

if __name__ == '__main__':
    app.run_server(debug=False)