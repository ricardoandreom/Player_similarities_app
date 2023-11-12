# libraries
import streamlit as st
from pathlib import Path
import app_functions as app
import lxml

# white Logo
white_logo_url = \
    "https://raw.githubusercontent.com/ricardoandreom/Data/master/Images/Personal%20Logos/Half%20Space%20Branco.png"

# black logo
black_logo_url = "https://raw.githubusercontent.com/ricardoandreom/Data/master/Images/Personal%20Logos/Half%20Space%20Preto.png"

# LANDING PAGE LINK
SOCIAL_MEDIA = {
    "Portfolio/Landing page": "https://ricardoandreom.github.io/ricardo_portfolio_page/"
}

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

############################################################################################

# STREAMLIT APP

# APP TITLE
st.title('Player similarities in Top-5 Leagues ⚽📈')

st.write('#')
# MAIN LOGO (WHITE)
twitter_url = 'https://twitter.com/HspaceAnalytics'
# MAIN LOGO (WHITE)
st.sidebar.markdown(f'<a href="{twitter_url}" target="_blank"><img src="{white_logo_url}" alt="Logo" style="width: 100%;"></a>', unsafe_allow_html=True)

# rest of the streamlit part
st.markdown("""
    <p style='font-size: 19px; font-weight: bold; color: white;'> 🔴 Choose a player and find out who are the 15th most
similar players, according to the implicit model. You can filter the results by age and league.</p>
""", unsafe_allow_html=True)

# GK note missing data
st.sidebar.write('Note: No data for goalkeepers yet!')

# portfolio url
st.sidebar.write("#")
cols = st.sidebar.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"**Made by @ricardoandreom 🚀⚽\n[{platform}]({link})**")

st.sidebar.markdown("""
    <p style='font-size: 26px; font-weight: bold; color: black;'>Inputs:</p>
""", unsafe_allow_html=True)

# creating the df
df = app.get_df_complete().reset_index()
max_value_sim = int(df['90s'].max())
min_ninety_sim = round(max_value_sim / 2)
df = df[(df['90s'] >= min_ninety_sim)]
df_orig = df.copy()

# choosing the player that the user wants to see similar players
player = st.sidebar.selectbox("**Name of the player:**", df_orig['Player'])
# fazer display da player_similars
st.title('Who are the most similar players to ' + player + '?')

############################################################################################
# SIMILARITIES PART

league_sim_options = ['La Liga', 'Bundesliga', 'Premier League', 'Serie A', 'Ligue 1', 'All']

default_league_index = league_sim_options.index('All')

league_sim = st.sidebar.selectbox("**League of the players:**", league_sim_options, index=default_league_index)

age_sim = st.sidebar.slider("**Max age of the players:**", 16, 40, value=38)

kofi_link = "https://ko-fi.com/ricardoandreom"
st.sidebar.markdown(f'<a href="{kofi_link}" target="_blank"><img src="https://cdn.ko-fi.com/cdn/kofi1.png" alt="Buy Me a Coffee" style="height: 40px;"></a>', unsafe_allow_html=True)
try:
    player_similars = app.similarity_process_v2(df_orig,player)[['Player',
                                                      '% similarity',
                                                      'Squad',
                                                      'Position',
                                                      'Secondary Position'
                                                      'Age',
                                                      '90s'
                                                      'League']]
    player_similars = player_similars[
        (player_similars['Age'] <= age_sim) &
        (player_similars['League'] == league_sim) if league_sim != 'All' else (player_similars['Age'] <= age_sim)].head(
        15)

    st.dataframe(player_similars.set_index('Player'))
except KeyError:
    st.warning("Error: We have no similarities for this player because this players has no enough minutes to get conclusions.")





