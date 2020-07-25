import requests
import json
import pprint
import numpy
import pandas as pd
#import xgboost as xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display
from bs4 import BeautifulSoup
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import re
# plt.style.use('ggplot')
# api_key = "RGAPI-d6357266-107b-47e5-bfab-ad0bc266863c"
# user_url = "https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/Novum?api_key="+api_key
# match_url = "https://na1.api.riotgames.com/lol/match/v4/matchlists/by-account/1L4ebzOq-2tERbxca_G36I6tfKXdfQtdM1ISAQGPMJ5sM04?api_key=" + api_key
# response = requests.get(user_url)
# response.json()

# ddragresponse = requests.get("http://ddragon.leagueoflegends.com/cdn/9.3.1/data/en_US/champion.json")
# ddragresponse.json()

# champRawData = json.loads(ddragresponse.text)
# crd = champRawData['data']
# kat_blurb = crd['Katarina']
# print(kat_blurb)
def getTeamData(teams_url):

    teams_page = requests.get(teams_url)

    #pp.pprint(teams_page.content)

    teams_soup = BeautifulSoup(teams_page.content,'html.parser')
    #print((teams_soup.prettify()).encode('utf8'))
    teamlist = teams_soup.find('table', class_='playerslist')

    teams_list = ["100 Thieves", "CLG", "Cloud9", "Dignitas","Evil Geniuses", "FlyQuest", "Golden Guardians", "Immortals", "Team Liquid", "TSM"]
    stats_list = ["Winrate", "Gd/min","Avg game time",  "Drag %", "GD at 15"]
    d = {}
    for team in teams_list:
        d[team] = []

        # print((teamlist.prettify()).encode('utf8'))
        # print()
    for team in teams_list:
        teamstats = teamlist.find('a',{'title' : team + ' stats'})
        stats = teamstats.find_all_next("td",limit = 21)
        #print(teamstats.text)
        counter = 0
        listcounter=0
        for data in stats:
            if(counter == 3 or counter == 7 or counter == 6 or counter == 15 or counter == 20):
                #print(stats_list[listcounter] + " "+ data.text)
                if('%' in data.text):
                    d[team].append(float(data.text[0:4]))
                elif ':' in data.text:
                    min_to_sec = float(data.text[0:2]) * 60
                    total = min_to_sec + float(data.text[3:5])
                    d[team].append(total)
                else:
                    d[team].append(float(data.text))
                listcounter+=1
            counter+=1
        #print(d)

    df = pd.DataFrame.from_dict(d, orient = 'index', columns = stats_list)
    # df = pd.DataFrame(numpy.random.randn(1000, 4), columns=['A','B','C','D'])
    #scatter_matrix(df, alpha=0.2)
    # pd.plotting.scatter_matrix(df, diagonal = 'kde')
    #plt.show()
    return df
        
    #print(teams_html)
def removePercent(my_str):
    index = my_str.find('%')
    new_str = ""
    for i in range(index):
        new_str= new_str +(my_str[i])
    return new_str
def getPlayerStats(playerName, year,champion ):
    player_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=PlayerByChampion&TS%5Byear%5D=" + str(year) + "&TS%5Bspl%5D=Yes&TS%5Blink%5D=" + playerName + "&pfRunQueryFormName=TournamentStatistics"
    player_page = requests.get(player_url)
    player_soup = BeautifulSoup(player_page.content,'html.parser')
    path = r'C:\Users\Basel\Desktop\coding\ml_sentdex\fiestaBot\playerstats.csv'
    mydf = pd.read_csv(path)
    row = mydf[mydf['Player']==playerName]
    
    goldpercentage=row['GOLD%']
   
    dmgpercentage = row['DMG%']
    
    goldpercentage = float(removePercent(goldpercentage.values[0]))/100
    dmgpercentage = float(removePercent(dmgpercentage.values[0]))/100
    dmg_per_gold = dmgpercentage/goldpercentage
    stats_list = ["Games", "WR", "KDA", "CS/M", "Gold/M","Dmg_per_gold" ] #gold %, damage %
    d = {}
    #print((player_soup.prettify()).encode('utf8'))
    table = player_soup.find(lambda tag: tag.name=='table' ) 
    rows = table.findAll(lambda tag: tag.name=='tr')
    for row in rows:
        if champion in row.text:
            cols = row.findAll(lambda tag: tag.name == 'td')
            counter = 0
            stats_counter = 0
            for col in cols:
                if(counter == 1 or counter == 4 or counter == 8 or counter == 10 or counter == 12):
                    
                    if('%' in col.text):
                        index = col.text.find('%')
                        d[stats_list[stats_counter]]=(float(col.text[0:index]))
                    else:
                        d[stats_list[stats_counter]]=(float(col.text)) 
                    stats_counter+=1
                
                counter+=1
    d[stats_list[5]] = dmg_per_gold
    return d
def getChampStats(champion,role):
    champ_url = "https://na.op.gg/champion/" + champion + "/statistics/" + role
    champ_page = requests.get(champ_url)
    champ_soup = BeautifulSoup(champ_page.content,'html.parser')
    winrateByLength = champ_soup.find(id="Trend-GameLengthWinRateGraph").next_siblings
    #print(champ_soup.encode('utf8'))
    #print()
    counter = 0
    winrates = []
    for sibling in winrateByLength:
        if counter == 1:
            strsib = str(sibling.encode('utf8'))
            start_substr = "\"y\":"
            yindex = strsib.find(start_substr)
            matches = re.finditer(start_substr, strsib)
            matches_positions = [match.start() for match in matches]
            for i in range(5):
                startindex = matches_positions[i]+4
                wr_num = ""
                temp = strsib[startindex]
                tempcount = 0
                while(temp != ',' and tempcount <5):
                    wr_num= wr_num+(strsib[startindex + tempcount])
                    temp = strsib[startindex+tempcount]
                    tempcount+=1
                winrates.append(str(wr_num))

        counter+=1
    return winrates

def appendAllStats(players, year, champions,roles):
    totalPlayerStats = pd.DataFrame()
    for i in range(len(players)):
        playerStats = getPlayerStats(players[i],year,champions[i])
        champStats = getChampStats(champions[i],roles[i])
        totalPlayerStats = totalPlayerStats.append(playerStats,ignore_index = True)
    return totalPlayerStats

teams_url = "https://gol.gg/teams/list/season-S10/split-Summer/region-NA/tournament-ALL/week-ALL/"
team_df = getTeamData(teams_url)
players = ["Licorice","Blaber","Nisqy","Zven","Vulcan%20(Philippe%20Laflamme)"]
champions = ["Sett","Olaf","Zoe","Ezreal","Thresh"]
roles = ["Top", "Jungle","Middle","adc","Support"]
#totalPlayerStats = appendAllStats(players,2020,champions, roles)
#getChampStats("Ashe","adc")
getPlayerStats("Licorice",2020,"Sett")
#print(totalPlayerStats)