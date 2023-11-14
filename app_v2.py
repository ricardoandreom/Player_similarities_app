import app_functions as app
import streamlit as st
import lxml
from pathlib import Path

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

st.title('Player similarities in Top-5 Leagues ⚽📈')

st.write('#')
twitter_url = 'https://twitter.com/HspaceAnalytics'
# MAIN LOGO (WHITE)
st.sidebar.markdown(f'<a href="{twitter_url}" target="_blank"><img src="{white_logo_url}" alt="Logo" style="width: 100%;"></a>', unsafe_allow_html=True)

# rest of the streamlit part
st.markdown("""
    <p style='font-size: 19px; font-weight: bold; color: white;'> 🔴 Choose a player and find out who are the most
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

df = app.get_df_complete().reset_index()
min_ninety_sim = round(df['90s'].max() / 2)
df = df[df['90s'] >= min_ninety_sim]

# choosing the player that the user wants to see similar players
player = st.sidebar.selectbox("**Name of the player:**", df['Player'])
# fazer display da player_similars
st.title('Who are the most similar players to ' + player + '?')

age_limit = st.slider("Max age:", min_value=16, max_value=40, value=40)
leagues = st.selectbox("Select the league:", ["All", "Liga 1", "Premier League", "La Liga", "Serie A", "Bundesliga"])
n = st.slider("Number of players displayed:", min_value=5, max_value=30, value=15)

kofi_link = "https://ko-fi.com/ricardoandreom"
st.sidebar.markdown(f'<a href="{kofi_link}" target="_blank"><img src="https://cdn.ko-fi.com/cdn/kofi1.png" alt="Buy Me a Coffee" style="height: 40px;"></a>', unsafe_allow_html=True)

try:
    df_final = app.final_df(player, age_limit=age_limit, leagues=leagues, n=n)
    st.dataframe(df_final[['Player', '% similarity', 'Position', 'Squad', 'League', 'Age', '90s']])
except KeyError:
    st.warning("We have no similarities for this player because this player has not enough minutes to draw conclusions. Try with another player.")


