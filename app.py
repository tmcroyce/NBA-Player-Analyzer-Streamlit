
# This idea and base code comes from 

import base64
from itertools import count
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import norm
from scipy.integrate import cumtrapz
plt.rcParams.update({'font.family':'Bell MT'})

st.title('NBA Player Analytics')

st.markdown("""
This app creates a full rundown of a basketball player. 
At least, hypothetically...
"""
)

st.sidebar.header('Inputs')
# load plyer data here... 
playerstats = pd.read_excel('data/2011-2021_trad_and_advanced_playerstats.xlsx')

#Sidebar - Team selection
unique_team = playerstats['Team'].unique()
ps_21 = playerstats.loc[playerstats['season']== 2021]
sorted_unique_team = sorted(unique_team)

unique_player = ps_21['PLAYER'].unique() # list of unique player names
sorted_unique_player = sorted(unique_player)

#selected_team = st.sidebar.multiselect('TEAM', sorted_unique_team, sorted_unique_team[:1])
selected_player = st.sidebar.multiselect('PLAYER', sorted_unique_player, sorted_unique_player[:1])

# -----------------------------  Player Image   ----------------------- ----------------
buncha_stats = pd.read_excel('data/merged_v3_with_photos.xlsx')
player_o3a =  buncha_stats[(buncha_stats['Player'].isin(selected_player))]
player_image = player_o3a.reset_index()
player_image = player_image.at[0,'path']
st.sidebar.image(player_image)

# filtering data
df_selected_player = playerstats[(playerstats['PLAYER'].isin(selected_player))]
keepem = ['season','PLAYER', 'Team', 'Age', 'GP', 'W', 'L','PTS', '3P%','FT%', 'Min', 'TS%', 'AST%', 'REB%', 'USG%', 'PACE']
df_selected_player = df_selected_player[keepem]

# ------------------------------- Size Breakdown     -----------------------------
size_df = pd.read_excel('data/NBA Player Combine Measurements.xlsx', sheet_name='final_measurements')
player_size_df = size_df[(size_df['PLAYER'].isin(selected_player))]
player_size_df = player_size_df.drop(columns = ['player_name']).set_index('PLAYER')

st.header('Display Player Stats of Selected Thangz')
st.write('Advanced Stats by Season')

df_selected_player = df_selected_player.sort_values('season', ascending = False).set_index('season')

st.dataframe(df_selected_player.style.format(precision=2, formatter={('MIN', 'TS%', 'AST%', 'REB%', 'USG%'): "{:.1f}"}))
st.header('Size Breakdown')
st.dataframe(player_size_df.style.format(precision=2))

# player sizing
player_size_df = player_size_df.reset_index()
player_height = player_size_df.at[0,'height_inches']
player_wingspan = player_size_df.at[0,'wingspan_inches'] 
player_weight = player_size_df.at[0,'weight_lbs']
player_position = player_size_df.at[0,'primary_position']

st.sidebar.subheader(f'Position: {player_position}')
st.sidebar.subheader(f'Height: {player_height}')
st.sidebar.subheader(f'Wingspan: {player_wingspan}')

# comparisons
sizeAdvanced_df = pd.read_excel('data/player_adv_2011-2022_and_sizes.xlsx')
recent_df = sizeAdvanced_df.loc[sizeAdvanced_df['season'] == 2021]
recent_df = recent_df.loc[recent_df['primary_position']== player_position]
recent_df = recent_df.loc[recent_df['weight_lbs']!= '-']
recent_df = recent_df.loc[recent_df['MIN'] >= 15]
recent_df = recent_df.dropna()
recent_df = recent_df.replace({'-','NaN'})
recent_df = recent_df.dropna()
recent_df = recent_df.drop_duplicates()
avg_height = recent_df['height_inches'].mean()
avg_wingspan = recent_df['wingspan_inches'].mean()
player = str(selected_player)[2:-2]                                # player stores the real name.

player_height_norm = round((player_height / avg_height -1) * 100,2)

# Writing analysis
if player_height_norm > 0:
    ovun = 'taller'
else:
    ovun = 'shorter'

player_wing_norm = round((player_wingspan / avg_wingspan -1) * 100,2)
if player_wing_norm > 0:
    ovun2 = 'taller'
else:  
    ovun2 = 'shorter'

player_size = round((player_height_norm + player_wing_norm)/2,2)
if player_size > 0:
    ovun3 = 'taller'
else:  
    ovun3 = 'shorter'

st.markdown(f'''{player}'s height is {player_height} inches compared to position average of {round(avg_height,2)} inches.
{player}'s height is {player_height_norm} percent {ovun} than the NBA position average. 

{player}'s wingspan is {player_wingspan} inches compared to the average of {round(avg_wingspan,2)} for his poisition.
{player}'s wingspan is {player_wing_norm} percent {ovun2} than the NBA position average. 

On average, {player} is {player_size} percent {ovun3} than the NBA position average.
''')

st.sidebar.subheader(f'Overall Player Size: {player_size} percent {ovun3} than position average')

recent_df['height_decile'] = pd.qcut(recent_df['height_inches'], 10, labels = False)
recent_df['wingspan_decile'] = pd.qcut(recent_df['wingspan_inches'], 10, labels = False)
player_height_decile = recent_df.loc[recent_df['PLAYER']== player]['height_decile'].values[0]
player_wingspan_decile = recent_df.loc[recent_df['PLAYER']== player]['wingspan_decile'].values[0] 

st.sidebar.subheader(f'Height Decile: {player_height_decile}')
st.sidebar.subheader(f'Wingspan Decile: {player_wingspan_decile}')
# ----------------------------  Plot: Size Comparison  -----------------------
df= recent_df
x = 'height_inches'
y = 'wingspan_inches'

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(df[x], df[y])

plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')

# Plot images
def getImage(path):
    return OffsetImage(plt.imread(path), zoom=.2, alpha = 1)

ab = AnnotationBbox(getImage(player_image),(player_height, player_wingspan))
ax.add_artist(ab)

plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
plt.vlines(df[x].mean(), df[y].min(), df[y].max(), color='#c2c1c0')
plt.title('Height vs Wingspan by Position', size = 18, fontweight='bold')
plt.ylabel('Wingspan', size = 12)
plt.xlabel('Height', size = 12)

st.pyplot(fig)

#   ----------------------------  Define some stuff  -----------------------

def player_df(df, cols_to_keep):
    data = df
    data = data.loc[data['Player'] == player]
    data = data.sort_values('season', ascending = False)
    data = data[cols_to_keep]
    return data

def player_position_avg(df):
    data = df
    data = data.loc[data['primary_position'] == player_position]
    data = data.loc[data['season'] == 2021]
    data = data.loc[data['MIN'] >= 15]
    return data

#player position scatter plot
def plot_pps(avg_position_df, x, y, title, xlabel, ylabel, player_df):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(avg_position_df[x], avg_position_df[y])
    plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
    plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')
    plt.title(title, size = 18, fontweight='bold')
    ab = AnnotationBbox(getImage(player_image),(player_df[x], player_df[y]))
    ax.add_artist(ab)
    plt.ylabel(ylabel, size = 12)
    plt.xlabel(xlabel, size = 12)
    st.pyplot(fig)

# barchart with averages of player position
def player_barChart_w_averageLine(df, x, y, chart_title, ylabel):
    data = df
    data = data.replace({'-','NaN'})
    pos_avg = data.loc[data['season'] >= 2021]
    pos_avg = pos_avg.loc[pos_avg['MIN'] >= 15]
    pos_avg = pos_avg.loc[pos_avg['primary_position']== player_position]
    pos_average = pos_avg[y].mean()
    pos_std = pos_avg[y].std()
    up_std = pos_average + pos_std
    player_data = data.loc[data['Player'] == player]
    # plot
    fig, ax= plt.subplots(figsize=(8, 6), dpi=200)
    plt.bar(player_data[x], player_data[y])
    plt.hlines(pos_average, xmin = min(player_data["season"]), xmax =max(player_data["season"]),  color='#c2c1c0', linestyles='dashed')
    plt.hlines(up_std, xmin = min(player_data["season"]), xmax =max(player_data["season"]),  color='#c2c1c0', linestyles='dashed')
    plt.xticks(np.arange(min(player_data["season"]),max(player_data["season"]+1),1))
    plt.xlabel('Season', size = 12)
    plt.ylabel(ylabel)
    plt.title(chart_title, size = 18, fontweight= 'bold')
    st.pyplot(fig)

# Function to plot a histogram for player position
def position_histogram(df, x, title, xlabel):
    data = df
    data = data.loc[data['primary_position'] == player_position]
    data = data.loc[data['season'] == 2021]
    data = data.loc[data['MIN'] >= 15]
    data = data.sort_values(x)
    data = data.reset_index(drop=True)
    # plot
    fig, ax= plt.subplots(figsize=(8, 6), dpi=200)
    plt.hist(data[x], bins=20)
    plt.title(title, size = 18, fontweight='bold')
    plt.xlabel(xlabel, size = 12)
    plt.ylabel('Count', size = 12)
        # Add CDF
    #st.pyplot(fig)
    #fig2, ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.hist(data[x], bins=20, cumulative=True, label='CDF', histtype='step', linewidth=1, color='#42241f')
    #plt.title('Cumulative Distribution Function', size = 18, fontweight='bold')
    #plt.yticks()
    st.pyplot(fig)

# Function to show selected player career averages
def player_career_averages(df, cols_to_show):
    data = df
    data = data.loc[data['Player'] == player]
    data = data.sort_values('season', ascending = False)
    data = data[cols_to_show]
    st.dataframe(data.style.format(precision=2))

# Function to plot player 2021 averages vs positional averages
def player_vs_position_scatter(df, x, y, xlabel, ylabel, title):
    data = df
    data = data.loc[data['season'] == 2021]
    playerdata = data.loc[data['Player'] == player]
    data = data.loc[data['MIN'] >= 15]
    data = data.loc[data['primary_position'] == player_position]
    # plot
    fig, ax= plt.subplots(figsize=(8, 6), dpi=200)
    ab = AnnotationBbox(getImage(player_image),(playerdata[x], playerdata[y]))
    ax.add_artist(ab)
    plt.scatter(data[x], data[y])
    plt.hlines(data[y].mean(), data[x].min(), data[x].max(), color='#c2c1c0')
    plt.vlines(data[x].mean(), data[y].min(), data[y].max(), color='#c2c1c0')
    plt.title(title, size = 18, fontweight='bold')
    plt.xlabel(xlabel, size = 12)
    plt.ylabel(ylabel, size = 12)
    st.pyplot(fig)

def positional_decile(df, x):
    data = df
    data = data.loc[data['primary_position'] == player_position]
    data = data.loc[data['season'] == 2021]
    data = data.loc[data['MIN'] >= 15]
    col = str(x) + '_decile'
    data[col] = pd.qcut(data[x], 10, labels=False)
    return data.loc[data['Player'] == player][col].values[0]


# -----------------------------------  Speed & Distance  -------------------------------------
st.header('Speed and Distance')
st.write(f'Players tend to slow down as they age. Is {player} slowing down?')
keep = ['season', 'Player', 'Avg\xa0Speed','Avg\xa0Speed\xa0Off','Avg\xa0Speed\xa0Def', 'Dist.\xa0Miles']
sd = pd.read_excel('data/tracking/withPos_0_2013-2022_speed_distance_player_data.xlsx')
sd = sd.loc[sd['MIN'] >= 15]
sd = sd.loc[sd['primary_position'] == player_position]
sd2 = player_df(sd, keep)
st.dataframe(sd2)     

selected_player_21 = sd2.loc[sd2['season'] == 2021]            # Player Most Recent Year
sd_avg = player_position_avg(sd)                                # Average position data

# ------- Plot OffSpeed vs DefSpeed 
plot_pps(sd_avg, 'Avg\xa0Speed\xa0Off', 'Avg\xa0Speed', 
                'Positional Speed, Offense vs Total','Offensive Speed Avg', 
                'Total Speed Avg', selected_player_21)


speed_decile = positional_decile(sd, 'Avg\xa0Speed')
st.sidebar.subheader(f'Speed Decile: {speed_decile}')

# -----------------------------------  Shooting -------------------------------------
st.header('Shooting Breakdown')
st.subheader('Open 3pt Shooting')
buncha_stats = pd.read_excel('data/merged_v3_with_photos.xlsx')
tokeep = ['Player', 'TEAM', 'season', 'tot_Open_3PM', 'tot_Open_3PA', 'Total_Open_3P%']

player_vs_position_scatter(buncha_stats, 'tot_Open_3PM', 'Total_Open_3P%', 'Open 3PM Per Game', 'Open 3P%', 'Open 3P% vs 3PM')
player_barChart_w_averageLine(buncha_stats, 'season', 'Total_Open_3P%', 'Open 3pt Shooting by Year', 'Open 3pt%')
player_career_averages(buncha_stats, tokeep)
position_histogram(buncha_stats, 'Total_Open_3P%', 'Position Open 3pt Shooting Distribution', 'Open 3pt%')


open_decile = positional_decile(buncha_stats, 'Total_Open_3P%')
st.sidebar.subheader(f'Open 3FG% Decile: {open_decile}')
# --------------------------------Contested Shooting --------------------------------
st.header('Contested Shooting')
st.write('''
Contested Shooting is defined as having a defender within 2-4 feet of the shot. 
This is more likely to occur the closer the player is to the basket, which may 
be important to remember. I.e., drives and jumpers are not apples to apples.
''')
to_keep = ['Player', 'TEAM', '2pt_2-4_FREQ', '2pt_2-4_2PM', '2pt_2-4_2PA', '2pt_2-4_FG%']
cont_shoot = buncha_stats
cont_shoot = cont_shoot.drop(cont_shoot.index[cont_shoot['2pt_2-4_FG%'] == '-'])
cont_shoot = cont_shoot.drop(cont_shoot.index[cont_shoot['2pt_2-4_FREQ'] == '-'])
cont_shoot['2pt_2-4_FG%'] = cont_shoot['2pt_2-4_FG%'].astype(np.float64)
cont_shoot['2pt_2-4_2PM'] = cont_shoot['2pt_2-4_2PM'].astype(np.float64)

player_career_averages(cont_shoot, to_keep)
player_vs_position_scatter(cont_shoot, '2pt_2-4_2PM', '2pt_2-4_FG%', 'Made Per Game', 'FG%', 'Contested 2pt% vs Frequency')
player_barChart_w_averageLine(cont_shoot, 'season', '2pt_2-4_FG%', 'Contested Shooting by Year', 'Contested 2pt%')
position_histogram(cont_shoot, '2pt_2-4_FG%', 'Position Contested 2pt Shooting Distribution', 'Contested 2pt%')

contested_decile = positional_decile(cont_shoot, '2pt_2-4_FG%')
st.sidebar.subheader(f'Contested 2FG% Decile: {contested_decile}')

# ---------------------------- Catch and Shoot Shooting --------------------------------- #
st.header('Catch and Shoot')
catchshoot = pd.read_excel('data/tracking/withPos_0_2013-2022_catch-shoot_player_data copy.xlsx')
catchshoot = catchshoot.drop(catchshoot.index[catchshoot['3P%'] == '-'])
catchshoot = catchshoot.drop(catchshoot.index[catchshoot['eFG%'] == '-'])
catchshoot['3P%'] = catchshoot['3P%'].astype(np.float64)
catchshoot['eFG%'] = catchshoot['eFG%'].astype(np.float64)

keep= ['season','Player', '3P%','eFG%', 'primary_position']
player_career_averages(catchshoot, keep)
player_vs_position_scatter(catchshoot, '3P%', 'eFG%', '3P%', 'eFG%', '3P% vs eFG%')
player_barChart_w_averageLine(catchshoot, 'season', '3P%', 'Catch and Shoot 3P% by Year', '3pt%')
position_histogram(catchshoot, '3P%', 'Position Catch and Shoot 3pt% Distribution', '3pt%')

catchshoot_decile = positional_decile(catchshoot, '3P%')
st.sidebar.subheader(f'Catch and Shoot 3FG% Decile: {catchshoot_decile}')

# --------------------------------Three Point Defense--------------------------------
st.header('Defense Breakdown')
st.header('Defensive 3PT%')

tokeep = ['Player', 'TEAM', 'season', 'DEF_3PT_FREQ' , 'DEF_3PT_DFGA',  'DEF_3PT_DFG%']
threept_defense = buncha_stats
threept_defense = threept_defense.drop(threept_defense.index[threept_defense['DEF_3PT_DFGA'] == '-'])
threept_defense = threept_defense.drop(threept_defense.index[threept_defense['DEF_3PT_FREQ'] == '-'])
threept_defense['DEF_3PT_DFG%'] = threept_defense['DEF_3PT_DFG%'].astype(np.float64)
threept_defense['DEF_3PT_FREQ'] = threept_defense['DEF_3PT_FREQ'].astype(np.float64)

player_career_averages(threept_defense, tokeep)
player_vs_position_scatter(threept_defense, 'DEF_3PT_FREQ', 'DEF_3PT_DFG%', 'Frequency', '3pt%', '3pt% vs Frequency')
player_barChart_w_averageLine(threept_defense, 'season', 'DEF_3PT_DFG%', '3pt Defense by Year', '3pt%')
position_histogram(threept_defense, 'DEF_3PT_DFG%', 'Position 3pt Defense Distribution', '3pt%')

threedee_decile = positional_decile(threept_defense, 'DEF_3PT_DFG%')
st.sidebar.subheader(f'3pt Defense 3FG% Decile: {threedee_decile}')

# ------------------------------ Defensive Field Goal %--------------------------------
st.header('Defensive FG%')
overall_d = pd.read_excel('data/tracking/withPos_0_2013-2022_defensive-impact_player_data.xlsx')
to_keep = ['season', 'Player', 'Team', 'DFG%', 'DFGA']
overall_d = overall_d.drop(overall_d.index[overall_d['DFG%'] == '-'])
overall_d['DFG%'] = overall_d['DFG%'].astype(np.float64)
overall_d = overall_d.drop(overall_d.index[overall_d['DFGA'] == '-'])
overall_d['DFGA'] = overall_d['DFGA'].astype(np.float64)

player_career_averages(overall_d, to_keep)
player_vs_position_scatter(overall_d, 'DFGA', 'DFG%', 'Defensive FG Attempts', 'DFG%', 'Defensive FG% vs FG Attempts')
player_barChart_w_averageLine(overall_d, 'season', 'DFG%', 'Defensive FG% by Year', 'Defensive FG%')
position_histogram(overall_d, 'DFG%', 'Position Defensive FG% Distribution', 'Defensive FG%')

deefg_decile = positional_decile(overall_d, 'DFG%')
st.sidebar.subheader(f'Defensive FG% Decile: {deefg_decile}')

# ---------------------------- Rebounding ------------------------------------------- #
st.header('Rebounding')
rebounds = buncha_stats
cols = ['season', 'Player', 'TEAM', 'REB%', 'DREB%']
player_career_averages(rebounds, cols)
player_vs_position_scatter(rebounds, 'REB%', 'DREB%', 'Rebound Percentage', 'Defensive Rebound Percentage', 'Rebounding vs Defensive Rebounds')
player_barChart_w_averageLine(rebounds, 'season', 'REB%', 'Rebounding by Year', 'Rebound%')
position_histogram(rebounds, 'REB%', 'Position Rebounding Distribution', 'Rebound%')

reb_decile = positional_decile(rebounds, 'REB%')
st.sidebar.subheader(f'Rebounding Decile: {reb_decile}')


# ---------------------------- Creating For Teammates --------------------------------- #
# AST% is The percentage of teammate field goals a player assisted on while they were on the floor
# AST to Pass% is  (Assists)/(Passes Made)
# AST Ratio - Assist Ratio is the number of assists a player averages per 100 possessions
# We should actually use AST Ratio. 
st.header('Creating for Teammates')
st.write('AST Ratio is the number of assists a player averages per 100 possessions')

player_passing = buncha_stats
keep= ['season','AST%', 'AST/TO', 'AST\xa0Ratio', 'Player', 'TEAM']
player_career_averages(player_passing, keep)
player_vs_position_scatter(player_passing, 'AST\xa0Ratio', 'AST/TO', 'Assist Ratio', 'AST/TO', 'Assist Percentage vs Assist/To')
player_barChart_w_averageLine(player_passing, 'season', 'AST\xa0Ratio', 'Assist Ratio by Year', 'Assist%')
position_histogram(player_passing, 'AST\xa0Ratio', 'Position Assist Ratio Distribution', 'Assist%')

ast_decile = positional_decile(player_passing, 'AST\xa0Ratio')
st.sidebar.subheader(f'Assist Ratio Decile: {ast_decile}')

# ---------------------------- Turnovers --------------------------------- #
st.header('Turnovers')
st.write('Turnover Percentage is the number of turnovers a player or team averages per 100 possessions used.')
keep= ['TO\xa0Ratio','AST\xa0Ratio', 'Player', 'TEAM', 'season']

player_career_averages(buncha_stats, keep)
player_vs_position_scatter(buncha_stats, 'TO\xa0Ratio', 'AST\xa0Ratio', 'Turnover Ratio', 'Assist Ratio', 'Turnover Ratio vs Assist Ratio')
player_barChart_w_averageLine(buncha_stats, 'season', 'TO\xa0Ratio', 'Turnover Ratio by Year', 'Turnover%')
position_histogram(buncha_stats, 'TO\xa0Ratio', 'Position Turnover Ratio Distribution', 'Turnover%')

turnover_decile = positional_decile(buncha_stats, 'TO\xa0Ratio')
st.sidebar.subheader(f'Turnover Ratio Decile: {turnover_decile}')
# ---------------------------- Matchup Breakdowns --------------------------------- #
option2 = st.sidebar.select_slider('Show Matchup Breakdown', options = ['No', 'Yes'])
if option2 == 'Yes':
    st.header('Player Matchups Last Year')
    st.subheader('Ranked by Player Points Per Play (Best Performances First)')
    matchupz = pd.read_csv('data/0_All_Matchups.csv')
    your_dude = matchupz[(matchupz['PLAYER'].isin(selected_player))]
    grp = your_dude.groupby('Defender')['PARTIALPOSS', 'MIN', 'PLAYERPTS', 'TEAMPTS', 'FGM', 'FGA', '3PM', '3PA', 'AST', 'TOV'].sum()
    grp['PlayerPtsPerPoss'] =grp['PLAYERPTS'] / grp['PARTIALPOSS']
    grp['TeamPtsPerPoss']= grp['TEAMPTS'] / grp['PARTIALPOSS']
    grp['3P%'] = grp['3PM'] / grp['3PA']
    grp['FG%'] = grp['FGM'] / grp['FGA']
    grp = grp.sort_values('PlayerPtsPerPoss', ascending = False)
    grp = grp.loc[grp['PARTIALPOSS'] >= 25] # At least 25 possessions in season
    st.dataframe(grp.style.format(precision=2))
    avg_grp = grp.mean()
    st.subheader('Player Average Matchup Performance')
    avg_grp2 = avg_grp.transpose()
    st.dataframe(avg_grp2.style.format(precision=2))

option1 = st.sidebar.select_slider('Show Playtype Breakdown', options = ['No', 'Yes'])
if option1 == 'Yes':
    # ----------------     Player Playtype      ------------------------- #
    st.header('Player Playtypes')
    # ----------------  Isolation -------------------#
    st.subheader('Isolation')
    playtypes = pd.read_excel('data/player_playtype_17-21.xlsx')
    iso = playtypes.loc[playtypes['play'] == 'isolation']
    player_iso = iso[(iso['Player'].isin(selected_player))]
    st.dataframe(player_iso.style.format(precision=2))

    # Compare to all NBA
    playtypes21 = iso.loc[iso['Season'] == '2021-22']
    player_21 = playtypes21[(playtypes21['Player'].isin(selected_player))]

    # plot 
    df= playtypes21
    x = 'Freq'
    y = 'PPP'
    fig, ax = plt.subplots(figsize=(8, 6), dpi=1200)
    ax.scatter(df[x], df[y])
    plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
    plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')

    # Plot images
    def getImage(path):
        return OffsetImage(plt.imread(path), zoom=.2, alpha = 1)
    ab = AnnotationBbox(getImage(player_image),(player_21[x], player_21[y]))
    ax.add_artist(ab)

    plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
    plt.vlines(df[x].mean(), df[y].min(), df[y].max(), color='#c2c1c0')
    plt.title('Points per Play vs Frequency, All NBA 2021', size = 22)
    plt.ylabel('Points Per Play', size = 16)
    plt.xlabel('Freqquency', size = 16)
    st.pyplot(fig)

    # ----------------  Pick and Roll: Ball Handler -------------------#
    st.subheader('Pick & Roll Ball Handler')
    bhandle = playtypes.loc[playtypes['play'] == 'ball-handler']
    player_bhandle = iso[(iso['Player'].isin(selected_player))]
    st.dataframe(player_bhandle.style.format(precision=2))

    # compare
    bhandle21 = bhandle.loc[bhandle['Season'] == '2021-22']
    player_bhandle_21 = bhandle21[(bhandle21['Player'].isin(selected_player))]

    # plot
    df= bhandle21
    x = 'Freq'
    y = 'PPP'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=1200)
    ax.scatter(df[x], df[y])

    plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
    plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')
    # Plot images
    def getImage(path):
        return OffsetImage(plt.imread(path), zoom=.2, alpha = 1)
    ab = AnnotationBbox(getImage(player_image),(player_bhandle_21[x], player_bhandle_21[y]))
    ax.add_artist(ab)

    plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
    plt.vlines(df[x].mean(), df[y].min(), df[y].max(), color='#c2c1c0')
    plt.title('Points per Play vs Frequency, All NBA 2021', size = 22)
    plt.ylabel('Points Per Play', size = 16)
    plt.xlabel('Freqquency', size = 16)
    st.pyplot(fig)

    # ----------------  Spot-Up Shooter -------------------#
    st.subheader('Spot-up Shooter')
    spup = playtypes.loc[playtypes['play'] == 'spot-up']
    spup = spup.dropna(subset = ['PPP', 'Freq'])
    player_spup = spup[(spup['Player'].isin(selected_player))]
    st.dataframe(player_spup.style.format(precision=2))

    # compare
    spup21 = spup.loc[spup['Season'] == '2021-22']
    player_spup_21 = spup21[(spup21['Player'].isin(selected_player))]

    # plot
    df= spup21
    x = 'Freq'
    y = 'PPP'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(df[x], df[y])

    plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
    plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')
    # Plot images
    ab = AnnotationBbox(getImage(player_image),(player_spup_21[x], player_spup_21[y]))
    ax.add_artist(ab)

    plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
    plt.vlines(df[x].mean(), df[y].min(), df[y].max(), color='#c2c1c0')
    plt.title('Points per Play vs Frequency, All NBA 2021', size = 22)
    plt.ylabel('Points Per Play', size = 16)
    plt.xlabel('Freqquency', size = 16)
    st.pyplot(fig)

    # ----------------  Pick and Roll: Roll-Man -------------------#
    st.subheader('Roll-Man')
    rollman = playtypes.loc[playtypes['play'] == 'roll-man']
    player_rollman = rollman[(rollman['Player'].isin(selected_player))]
    player_rollman = player_rollman.sort_value('season')
    player_rollman = player_rollman.set_index('season')
    st.dataframe(player_rollman.style.format(precision=2))

    # compare
    rollman21 = rollman.loc[rollman['Season'] == '2021-22']
    player_rollman_21 = rollman21[(rollman21['Player'].isin(selected_player))]

    # plot
    df= rollman21
    x = 'Freq'
    y = 'PPP'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(df[x], df[y])

    plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
    plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')
    # Plot images
    ab = AnnotationBbox(getImage(player_image),(player_rollman_21[x], player_rollman_21[y]))
    ax.add_artist(ab)

    plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
    plt.vlines(df[x].mean(), df[y].min(), df[y].max(), color='#c2c1c0')
    plt.title('Points per Play vs Frequency, All NBA 2021', size = 22)
    plt.ylabel('Points Per Play', size = 16)
    plt.xlabel('Freqquency', size = 16)
    st.pyplot(fig)

    # ----------------  Post-up -------------------#
    st.subheader('Post-Up')
    postup = playtypes.loc[playtypes['play'] == 'playtype-post-up']
    player_postup = postup[(postup['Player'].isin(selected_player))]
    st.dataframe(player_postup.style.format(precision=2))

    # compare
    postup21 = postup.loc[postup['Season'] == '2021-22']
    player_postup_21 = postup21[(postup21['Player'].isin(selected_player))]

    # plot
    df= postup21
    x = 'Freq'
    y = 'PPP'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(df[x], df[y])

    plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
    plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')
    # Plot images
    ab = AnnotationBbox(getImage(player_image),(player_postup_21[x], player_postup_21[y]))
    ax.add_artist(ab)

    plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
    plt.vlines(df[x].mean(), df[y].min(), df[y].max(), color='#c2c1c0')
    plt.title('Points per Play vs Frequency, All NBA 2021', size = 22)
    plt.ylabel('Points Per Play', size = 16)
    plt.xlabel('Freqquency', size = 16)
    st.pyplot(fig)


# TODO: Add: 
# Injury report (injuries through years), 
# position estimate (basketball-reference),
# Advanced stats from basketball-reference
# Play-by-play (thats what they call it, its not actually) from basketball reference