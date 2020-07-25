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
plt.style.use('ggplot')
# api_key = "RGAPI-d211d240-d0a2-42df-8564-1aa5ac451612"
# user_url = "https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/Novum?api_key="+api_key
# match_url = "https://na1.api.riotgames.com/lol/match/v4/matchlists/by-account/1L4ebzOq-2tERbxca_G36I6tfKXdfQtdM1ISAQGPMJ5sM04?api_key=" + api_key
# response = requests.get(user_url)
# response.json()

# ddragresponse = requests.get("http://ddragon.leagueoflegends.com/cdn/9.3.1/data/en_US/champion.json")
# ddragresponse.json()

# champRawData = json.loads(ddragresponse.text)
# crd = champRawData['data']
# kat_blurb = crd['Katarina']['blurb']
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
    scatter_matrix(df, alpha=0.2)
    # pd.plotting.scatter_matrix(df, diagonal = 'kde')
    plt.show()
    return df
        
    #print(teams_html)


teams_url = "https://gol.gg/teams/list/season-S10/split-Summer/region-NA/tournament-ALL/week-ALL/"
df = getTeamData(teams_url)
print(df)