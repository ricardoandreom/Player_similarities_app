# libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import lxml

# creating dataframe function
def get_df(url_df, columns_remaining_list):
    df = pd.read_html(url_df)[0]
    df.columns = [' '.join(col).strip() for col in df.columns]
    df = df.reset_index(drop=True)

    # creating a list with new names
    new_columns = []
    for col in df.columns:
        if 'level_0' in col:
            new_col = col.split()[-1]  # takes the last name
        else:
            new_col = col
        new_columns.append(new_col)

    # rename columns
    df.columns = new_columns
    df = df.fillna(0)

    if 'Playing Time 90s' in new_columns:
        df = df.rename(columns={'Playing Time 90s': '90s'})
    else:
        df = df.rename(columns={'90s': '90s'})

    df = df[['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', '90s'] +
            columns_remaining_list]

    df['Age'] = df['Age'].str[:2]
    df['Position_2'] = df['Pos'].str[3:]
    df['Position'] = df['Pos'].str[:2]
    df['Nation'] = df['Nation'].str.split(' ').str.get(1)
    df['League'] = df['Comp'].str.split(' ').str.get(1)
    df['League_'] = df['Comp'].str.split(' ').str.get(2)
    df['League'] = df['League'] + ' ' + df['League_']
    df = df.drop(columns=['League_', 'Comp', 'Rk', 'Pos', 'Born'])

    df['Position'] = df['Position'].replace({'MF': 'Midfielder', 'DF': 'Defender', 'FW': 'Forward', 'GK': 'Goalkeeper'})
    df['Position_2'] = df['Position_2'].replace({'MF': 'Midfielder', 'DF': 'Defender',
                                                 'FW': 'Forward', 'GK': 'Goalkeeper'})
    df['League'] = df['League'].fillna('Bundesliga')

    return df


# function to edit the df resulted by the concatenation
def edit_df(df_concatenated):
    df_concatenated = df_concatenated.rename(columns={'Per 90 Minutes G-PK': 'Non-penalty goals',
                                                      'SCA SCA': 'Shot creation actions',
                                                      'Aerial Duels Won%': 'Aerial duels won %',
                                                      'Performance Fld': 'Fouls won',
                                                      'Performance Crs': 'Crosses',
                                                      'Performance Recov': 'Ball recoveries',
                                                      'Expected npxG': 'npxG',
                                                      'Expected npxG/Sh': 'npxG/shot',
                                                      'Standard SoT': 'Shots on target',
                                                      'Total Cmp': 'Passes completed',
                                                      'KP': 'Key passes',
                                                      '1/3': 'Final 3rd passes',
                                                      'Long Att': 'Long passes',
                                                      'PrgP': 'Progressive passes',
                                                      'Total Cmp%': 'Pass completion %',
                                                      'Tkl+Int': 'Tackles + interceptions',
                                                      'Blocks Sh': 'Shots blocked',
                                                      'Clr': 'Clearances',
                                                      'Take-Ons Att': 'Dribbles attempted',
                                                      'Take-Ons Succ%': 'Dribbles success %',
                                                      'Carries CPA': 'Carries into penalty area',
                                                      'Receiving Rec': 'Passes received',
                                                      'Receiving PrgR': 'Progressive passes received',
                                                      'Carries PrgC': 'Progressive carries'
                                                      })

    numeric_columns = ['Age', '90s', 'Non-penalty goals', 'Shot creation actions', 'Aerial duels won %', 'Fouls won',
                       'Crosses', 'Ball recoveries', 'npxG', 'npxG/shot', 'Shots on target', 'Passes completed',
                       'Key passes', 'Expected xA', 'Final 3rd passes', 'Long passes', 'Progressive passes', 'Pass completion %',
                       'Tackles + interceptions', 'Shots blocked', 'Clearances', 'Dribbles attempted',
                       'Dribbles success %', 'Carries into penalty area', 'Passes received',
                       'Progressive passes received', 'Progressive carries']

    for j in numeric_columns:
        df_concatenated[j] = pd.to_numeric(df_concatenated[j], errors='coerce')

    ninety_columns = ['Non-penalty goals', 'Shot creation actions', 'Fouls won',
                      'Crosses', 'Ball recoveries', 'npxG', 'Shots on target',
                      'Passes completed', 'Key passes', 'Expected xA', 'Final 3rd passes',
                      'Long passes', 'Progressive passes', 'Tackles + interceptions',
                      'Shots blocked', 'Clearances', 'Dribbles attempted',
                      'Carries into penalty area', 'Passes received',
                      'Progressive passes received', 'Progressive carries']
    # age filter
    df_concatenated = df_concatenated[df_concatenated['90s'] > 0]

    for i in ninety_columns:
        df_concatenated[i] = (df_concatenated[i] / df_concatenated['90s']).round(2)

    return df_concatenated


def get_df_complete():
    url_standard = 'https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats#stats_standard'
    url_shotcreation = 'https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats#stats_gca'
    url_misc = 'https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats#stats_misc'
    url_shoot = 'https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats#stats_shooting'
    url_passing = 'https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats#stats_passing'
    url_def = 'https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats#stats_defense'
    url_poss = 'https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats#stats_possession'

    df_standard = get_df(url_standard, ['Per 90 Minutes G-PK'])
    df_shotcreation = get_df(url_shotcreation, ['SCA SCA'])
    df_misc = get_df(url_misc, ['Aerial Duels Won%', 'Performance Fld', 'Performance Crs', 'Performance Recov'])
    df_shoot = get_df(url_shoot, ['Expected npxG', 'Expected npxG/Sh', 'Standard SoT'])
    df_passing = get_df(url_passing, ['Total Cmp', 'KP', 'Expected xA', '1/3', 'Long Att', 'PrgP', 'Total Cmp%'])
    df_def = get_df(url_def, ['Tkl+Int', 'Blocks Sh', 'Clr'])
    df_poss = get_df(url_poss, ['Take-Ons Att', 'Take-Ons Succ%', 'Carries CPA', 'Receiving Rec', 'Receiving PrgR',
                                'Carries PrgC'])

    # Define a list of dataframes
    dfs = [df_shotcreation, df_misc, df_shoot, df_passing, df_def, df_poss]

    # Loop through the list and drop the columns
    for df in dfs:
        df.drop(columns=['Player', 'Squad', 'Age', 'Nation', 'Position', 'League', '90s', 'Position_2'], inplace=True)

    # concatenate all dat
    frames = [df_standard, df_shotcreation, df_misc, df_shoot, df_passing, df_def, df_poss]
    df = pd.concat(frames, axis=1, join='outer')

    ########################################################################################################

    df = edit_df(df)

    df = df[df['Position'] != 'Goalkeeper']

    return df


def name_abrev(row):
    palavras = row.split()
    if len(palavras) == 3:
        return ' '.join(palavras[1:])
    elif len(palavras) > 0:
        return palavras[0][0] + '.' + ' ' + palavras[1] if len(palavras) > 1 else palavras[0]


def reduced_df(df):
    # Removing non numerical columns
    df = df.drop(columns=['Nation', 'Squad', 'Position_2', 'Position', 'League'])
    player_names = df['Player'].tolist()
    df = df.drop(['Player'], axis=1)

    x = df.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)

    pca = PCA(n_components=2)
    reduced = pd.DataFrame(pca.fit_transform(X_norm))

    return reduced


def wcss_k_opt(reduced):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(reduced)
        wcss.append(kmeans.inertia_)

    # plot the number of clusters graph if ypu want
    # plt.plot(range(1, 11), wcss)
    # plt.xlabel('Number of clusters (K)')
    # plt.ylabel('WCSS')

    # Calculating the elbow point
    x = np.array(range(1, 11))
    y = np.array(wcss)
    n = len(x)
    A = np.array([x, np.ones(n)])
    w = np.linalg.lstsq(A.T, y, rcond=-1)[0]
    yy = w[0] * x + w[1]
    # interception with the previous plot to define the optimal number of clusters
    # plt.plot(x, yy, '-r')

    # Finding the optimal number of clusters
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    k_opt = np.argmax(diff_r) + 2
    # print("Optimal number of clusters:", k_opt)

    return k_opt


def clustering_reduced_col_concatenation(reduced, df, k_opt):
    kmeans = KMeans(n_clusters=k_opt)
    kmeans = kmeans.fit(reduced)

    labels = kmeans.predict(reduced)
    clusters = kmeans.labels_.tolist()

    player_names = df['Player'].tolist()

    reduced['cluster'] = clusters
    reduced['Player'] = player_names
    reduced.columns = ['x', 'y', 'cluster', 'Player']

    reduced = pd.merge(reduced, df[['Player', 'Squad', 'League', 'Age', '90s', 'Nation']], on='Player', how='outer')

    return reduced


def cluster_process(df):
    reduced = reduced_df(df)
    k_opt = wcss_k_opt(reduced)
    reduced_final = clustering_reduced_col_concatenation(reduced, df, k_opt)

    return reduced_final


def distance_matrix_reduced_calculation(reduced_sim):
    distance_matrix = pd.DataFrame(index=reduced_sim['Player'], columns=reduced_sim['Player'])
    for i in range(len(distance_matrix)):
        x_i = reduced_sim.iloc[i, 0]
        y_i = reduced_sim.iloc[i, 1]
        for j in range(len(distance_matrix)):
            x_j = reduced_sim.iloc[j, 0]
            y_j = reduced_sim.iloc[j, 1]
            distance_matrix.iloc[i, j] = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)

    return distance_matrix


def similarity_matrix(reduced_sim, distance_matrix):
    similarity_matrix = pd.DataFrame(index=reduced_sim['Player'], columns=reduced_sim['Player'])
    max_euc_distance = distance_matrix.max(axis=1)
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            similarity_matrix.iloc[i, j] = ((max_euc_distance[i]-distance_matrix.iloc[i, j])*100/max_euc_distance[i])

    return similarity_matrix


def similarity_process(df, player):
    reduced_sim = cluster_process(df)
    distance_matrix = distance_matrix_reduced_calculation(reduced_sim)
    sim_matrix = similarity_matrix(reduced_sim, distance_matrix)

    # Exemplo de como obter os jogadores mais similares ao jogador "player1"
    similarities = sim_matrix.loc[player].sort_values(ascending=False)
    player_similars = similarities.reset_index()
    player_similars.rename(columns={player: '% similarity'}, inplace=True)
    player_similars['% similarity'] = player_similars['% similarity'].apply(lambda x: round(x.item(), 2))
    player_similars = player_similars.drop(0)

    player_similars = player_similars.merge(df[['Player', 'Squad', 'League', 'Age', 'Nation', '90s', 'Position']],
                                            on='Player', how='outer')
    player_similars['Age'] = player_similars['Age'].astype(int)

    return player_similars


def calculate_euclidean_distance_v2(reduced_sim):
    features = reduced_sim[['x', 'y']].values

    distance_matrix = cdist(features, features, metric='euclidean')

    distance_df = pd.DataFrame(distance_matrix, index=reduced_sim['Player'], columns=reduced_sim['Player'])

    return distance_df

def calculate_similarity_v2(distance_matrix):
    similarity_matrix = 1 / (1 + distance_matrix)
    np.fill_diagonal(similarity_matrix.values, 0)
    return similarity_matrix * 100


def similarity_process_v2(df, player):
    reduced_sim = cluster_process(df)
    distance_matrix = calculate_euclidean_distance_v2(reduced_sim)
    sim_matrix = calculate_similarity_v2(distance_matrix)

    # Exemplo de como obter os jogadores mais similares ao jogador "player1"
    similarities = sim_matrix.loc[player].sort_values(ascending=False)
    player_similars = similarities.reset_index().drop_duplicates()
    player_similars.rename(columns={player: '% similarity'}, inplace=True)
    player_similars['% similarity'] = player_similars['% similarity'].round(2)

    df_sorted = df.sort_values(by='90s', ascending=False)
    result_df = df_sorted.drop_duplicates(subset='Player')

    player_similars = player_similars.merge(result_df[['Player', 'Squad', 'League',
                                                'Age', 'Nation', '90s', 'Position']],
                                                 on='Player', how='left')
    player_similars['Age'] = player_similars['Age'].astype(int)

    return player_similars

def cosine_sim_player(player_name):
    df = get_df_complete()

    index_player = {name: index for index, name in enumerate(df['Player'])}

    index = index_player[player_name]

    scaler = StandardScaler()

    data_players = df.drop(columns=['Player','Nation', 'Squad',
                                     'Age','Position','Position_2',
                                     'League'])

    data_players_normalized = scaler.fit_transform(data_players)

    player_data = data_players.iloc[index].values.reshape(1, -1)
    player_data_normalized = scaler.transform(player_data)
    similarities = cosine_similarity(player_data_normalized, data_players_normalized)
    similarities_converted = ((similarities + 1) / 2 * 100).round(1)

    result_df = pd.DataFrame({
        'Player': df['Player'],
        'Squad': df['Squad'],
        '% similarity': similarities_converted[0]
    })

    return result_df.sort_values(by='% similarity', ascending=False)

def final_df(player, age_limit=45, leagues='All',n=15):
    df = get_df_complete()
    player_similars = cosine_sim_player(player)[1:]
    player_similars = player_similars.merge(df, on=['Player','Squad'], how='left')

    if leagues != 'All':
        player_similars = player_similars[(player_similars['League'] == leagues) & (player_similars['Age'] <= age_limit)
        ].head(n)
    else:
        player_similars = player_similars[
            player_similars['Age'] <= age_limit].head(n)

    df_final = player_similars[1:].round(2)
    df_final['Age'] = df_final['Age'].astype(int)
    df_final = df_final.rename(columns={'Position_2': 'Secondary Position'})

    return df_final
