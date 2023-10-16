# libraries
import pandas as pd
import streamlit as st
from pathlib import Path
import app_functions as app
import lxml

# white Logo
white_logo_url = \
    "https://raw.githubusercontent.com/ricardoandreom/Liga-Portugal-2021-22/master/Half%20Space%20Branco.png"

# black logo
black_logo_url = "https://raw.githubusercontent.com/ricardoandreom/Liga-Portugal-2021-22/master/halfspace_analytics.png"

# LANDING PAGE LINK
SOCIAL_MEDIA = {
    "Portfolio/Landing page": "https://ricardo-marques-digital-cv.streamlit.app/"
}

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

############################################################################################

# creating the df
df = app.get_df_complete()
df_orig = df.copy()

############################################################################################

# STREAMLIT APP

# APP TITLE
st.title('Player similarities in Top-5 Leagues ⚽📈')

st.write('#')
# MAIN LOGO (WHITE)
st.sidebar.image(white_logo_url, use_column_width=True)

st.sidebar.markdown("""
    <p style='font-size: 26px; font-weight: bold; color: black;'>Player similarity inputs:</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <p style='font-size: 16px; font-weight: bold; color: black;'>Assess a player level of similarity of with players in the same position in Top-5 Leagues:</p>
""", unsafe_allow_html=True)

# choosing the player that the user wants to see similar players
player = st.sidebar.selectbox("**Name of the player:**", df_orig['Player'])

# GK note missing data
st.sidebar.write('Note: No data for goalkeepers yet!')

############################################################################################
# SIMILARITIES PART

player_position_sim = df_orig.set_index('Player').loc[player]['Position']

'''
# define the max value that the user can choose for min 90s played
if df_orig['90s'].max() == 1:
    max_value_sim = int(df_orig['90s'].max())
elif df_orig['90s'].max() < 5:
    max_value_sim = int(df_orig['90s'].max()) - 1
elif df_orig['90s'].max() < 8:
    max_value_sim = int(df_orig['90s'].max()) - 3
elif df_orig['90s'].max() < 12:
    max_value_sim = int(df_orig['90s'].max()) - 4
else:
    max_value_sim = int(df_orig['90s'].max()) - 7
'''

max_value_sim = int(df['90s'].max()) # - 5)

min_ninety_sim = st.sidebar.slider("**Select mininum 90s played you want to consider for the model similarities:**", 1, max_value_sim, key="min_ninety_sim")

df_orig = df_orig[(df_orig['Position'] == player_position_sim) & (df_orig['90s'] >= min_ninety_sim)]

player_similars = app.similarity_process(df_orig,player)

league_sim = st.sidebar.selectbox("**League of the players:**", ['La Liga', 'Bundesliga', 'Premier League', 'Serie A', 'Ligue 1', 'All'])
age_sim = st.sidebar.selectbox("**Max age of the players:**", player_similars['Age'])


if league_sim != 'All':
    player_similars = player_similars[(player_similars['League'] == league_sim) & (player_similars['Age'] <= age_sim)
    ].head(15)
else:
    player_similars = player_similars[
        player_similars['Age'] <= age_sim].head(15)

#########################################################################################################################

# rest of the streamlit part
st.markdown("""
    <p style='font-size: 19px; font-weight: bold; color: white;'> 🔴 Choose a player and find out who are the 15 most
    similar players, according to the implicit model. You can filter the results per age, league and mininum 90s played.</p>
""", unsafe_allow_html=True)

# fazer display da player_similars
st.title('Who are the most similar players to ' + player + '?')
st.dataframe(player_similars.set_index('Player'))

#####################################################

# portfolio url
st.sidebar.write("#")
cols = st.sidebar.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"**Made by @ricardoandreom 🚀⚽\n[{platform}]({link})**")

