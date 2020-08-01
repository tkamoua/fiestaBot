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
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm
from time import time 
from sklearn.metrics import f1_score
#returns a dataframe of a team's winrate, gd/min, avg game time, drag %, gd at 15
def getTeamData(team):
    teams_url = "https://gol.gg/teams/list/season-S10/split-Summer/region-NA/tournament-ALL/week-ALL/"
    teams_page = requests.get(teams_url)

    #pp.pprint(teams_page.content)

    teams_soup = BeautifulSoup(teams_page.content,'html.parser')
    #print((teams_soup.prettify()).encode('utf8'))
    teamlist = teams_soup.find('table', class_='playerslist')

    
    stats_list = ["Winrate", "Gd/min","Avg game time",  "FT %", "Drag %", "GD at 15"]
    d = {}
    
    d[team] = []

        # print((teamlist.prettify()).encode('utf8'))
        # print()
    
    teamstats = teamlist.find('a',{'title' : team + ' stats'})
    stats = teamstats.find_all_next("td",limit = 21)
    #print(teamstats.text)
    counter = 0
    listcounter=0
    for data in stats:
        if(counter == 3 or counter == 7 or counter == 6 or counter == 15 or counter == 13 or counter == 20):
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

    #df = pd.DataFrame.from_dict(d, orient = 'index', columns = stats_list)
    # df = pd.DataFrame(numpy.random.randn(1000, 4), columns=['A','B','C','D'])
    #scatter_matrix(df, alpha=0.2)
    # pd.plotting.scatter_matrix(df, diagonal = 'kde')
    #plt.show()
    return d

#removes the percent sign from a string 
def removePercent(my_str):
    index = my_str.find('%')
    new_str = ""
    for i in range(index):
        new_str= new_str +(my_str[i])
    return new_str
#returns dictionary of player # games, wr, kda, cs/min, gold/min, dmg per gold %
def getPlayerStats(playerName, year,champion,playername_oracle, isSpring ):
    #print("Name: " +playerName)

    if(playerName =="Vulcan"):
        playerName = "Vulcan%20(Philippe%20Laflamme)"
    elif playerName == "Closer":
        playerName = "Closer%20(Can%20Çelik)"
    elif playerName == "Apollo":
        playerName = "Apollo%20(Apollo%20Price)"
    elif playerName == "Wind":
        playerName = "Wind%20(Oh%20Myeong-jin)"
    elif playerName == "Solo":
        playerName = "Solo%20(Colin%20Earnest)"
    elif playerName == "Deus":
        playerName = "Deus%20(Alexey%20Zatorski)"
    spring_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bshowrecentgame%5D=True&TS%5Bpreload%5D=PlayerByChampion&TS%5Btournament%5D=LCS%202020%20Spring&TS%5Blink%5D="+playerName+"&pfRunQueryFormName=TournamentStatistics"
    springplayoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bshowrecentgame%5D=True&TS%5Bpreload%5D=PlayerByChampion&TS%5Btournament%5D=LCS%202020%20Spring%20Playoffs&TS%5Blink%5D="+playerName+"&pfRunQueryFormName=TournamentStatistics"
    player_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=PlayerByChampion&TS%5Byear%5D=" + str(year) + "&TS%5Bspl%5D=Yes&TS%5Blink%5D=" + playerName + "&pfRunQueryFormName=TournamentStatistics"
    
    player_page = requests.get(player_url)
    spring_page = requests.get(spring_url)
    springplayoffs_page = requests.get(springplayoffs_url)
    springplayoffs_soup = BeautifulSoup(springplayoffs_page.content,'html.parser')
    player_soup = BeautifulSoup(player_page.content,'html.parser')
    spring_soup = BeautifulSoup(spring_page.content,'html.parser')
    path = r'C:\Users\Basel\Desktop\coding\ml_sentdex\fiestaBot\playerstats.csv'
    mydf = pd.read_csv(path)
    row = mydf[mydf['Player']==playername_oracle]
    goldpercentage = 1
    dmgpercentage = 1
    if row.size != 0:
        goldpercentage=row['GOLD%']
        dmgpercentage = row['DMG%']
        goldpercentage = float(removePercent(goldpercentage.values[0]))/100
        dmgpercentage = float(removePercent(dmgpercentage.values[0]))/100
    dmg_per_gold = dmgpercentage/goldpercentage
    stats_list = ["Player","Champion","Games", "WR", "KDA", "CS/M", "Gold/M","Dmg_per_gold" ] #gold %, damage %
    d = {}
    if isSpring == False:
        #print((player_soup.prettify()).encode('utf8'))
        table = player_soup.find(lambda tag: tag.name=='table' ) 
        rows = table.findAll(lambda tag: tag.name=='tr')
        
        d[stats_list[0]]= playername_oracle
        d[stats_list[1]]= champion
        for row in rows:
            if champion == "Vel'koz":
                champion = "Vel'Koz"
            if champion == "Cho'gath":
                champion = "Cho'Gath"
            if champion == "Kha'zix":
                champion = "Kha'Zix"
            if champion in row.text:
                cols = row.findAll(lambda tag: tag.name == 'td')
                counter = 0
                stats_counter = 2
                for col in cols:
                    if(counter == 1 or counter == 4 or counter == 8 or counter == 10 or counter == 12):
                        
                        if('%' in col.text):
                            index = col.text.find('%')
                            d[stats_list[stats_counter]]=(float(col.text[0:index]))
                        else:
                            d[stats_list[stats_counter]]=(float(col.text)) 
                        stats_counter+=1
                    
                    counter+=1
        d[stats_list[7]] = dmg_per_gold
    
    else:
        #print((player_soup.prettify()).encode('utf8'))
        springtable = spring_soup.find(lambda tag: tag.name=='table' ) 
        springrows = springtable.findAll(lambda tag: tag.name=='tr')
        springplayoffstable = springplayoffs_soup.find(lambda tag: tag.name=='table' )
        springplayoffsrows = springplayoffstable.findAll(lambda tag: tag.name=='tr')
        d[stats_list[0]]= playername_oracle
        d[stats_list[1]]= champion
        foundchamp = False
        for row in springrows:
            if champion == "Vel'koz":
                champion = "Vel'Koz"
            if champion == "Cho'gath":
                champion = "Cho'Gath"
            if champion == "Kha'zix":
                champion = "Kha'Zix"
            if champion in row.text:
                cols = row.findAll(lambda tag: tag.name == 'td')
                counter = 0
                stats_counter = 2
                for col in cols:
                    if(counter == 1 or counter == 4 or counter == 8 or counter == 10 or counter == 12):
                        
                        if('%' in col.text):
                            index = col.text.find('%')
                            d[stats_list[stats_counter]]=(float(col.text[0:index]))
                        else:
                            d[stats_list[stats_counter]]=(float(col.text)) 
                        stats_counter+=1
                    
                    counter+=1
                foundchamp = True
                
        if foundchamp == False:
            for row in springplayoffsrows:
                if champion == "Vel'koz":
                    champion = "Vel'Koz"
                if champion == "Cho'gath":
                    champion = "Cho'Gath"
                if champion == "Kha'zix":
                    champion = "Kha'Zix"
                if champion in row.text:
                    cols = row.findAll(lambda tag: tag.name == 'td')
                    counter = 0
                    stats_counter = 2
                    for col in cols:
                        if(counter == 1 or counter == 4 or counter == 8 or counter == 10 or counter == 12):
                            
                            if('%' in col.text):
                                index = col.text.find('%')
                                d[stats_list[stats_counter]]=(float(col.text[0:index]))
                            else:
                                d[stats_list[stats_counter]]=(float(col.text)) 
                            stats_counter+=1
                        
                        counter+=1
                    foundchamp = True
                    
        

        d[stats_list[7]] = dmg_per_gold
    return d
#returns a dictionary of champion winrate in role at min 0-25, 25-30, 30-35, 35-40, 40+
def getAllChampStats():
    ddragresponse = requests.get("http://ddragon.leagueoflegends.com/cdn/9.3.1/data/en_US/champion.json")
    ddragresponse.json()

    champRawData = json.loads(ddragresponse.text)
    crd = champRawData['data']
    
    counter = 0
    for champ in crd:
        counter+=1
    champ_arr = numpy.empty([counter,1],dtype = object)
    index = 0
    for champ in crd:
        champ_arr[index] = champ
        index +=1
    return champ_arr
       
# print(kat_blurb)
def getChampStatsGG(champion):
    champ_url = "https://champion.gg/champion/" + champion
    champ_page = requests.get(champ_url)
    champ_soup = BeautifulSoup(champ_page.content,'html.parser')
    searched_word = "gameLength"
    results = champ_soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word)), recursive=True)
    #print ('Found the word "{0}" {1} times\n'.format(searched_word, len(results)))
    wrs = {}
    times = ["0-25","25-30","30-35","35-40","40+"]
    #print(champion)
    
    for content in results:
        #words = content.split()
        
        index = content.find(searched_word)
        # if champion == "Aphelios":
        #     print(content[index:])
        firstindex = index+14
        firstindex_end = firstindex+5
        if content[firstindex-1] != '"':
            firstindex = firstindex-1
            firstindex_end = firstindex+2
        elif content[firstindex] == '0':
            firstindex_end = firstindex+4
        elif content[firstindex+2] != '.':
                firstindex_end = firstindex+2
        
                
        first = content[firstindex:firstindex_end]
        wrs[times[0]] = (float(first))
        if content[index+13] != '"':
            firstindex_end = firstindex_end - 1
        #print(content[index:])
        
       # print(first)

        secondindex = firstindex_end+3
        secondindex_end = secondindex + 5
        if content[secondindex+2] != '.':
            secondindex_end = secondindex+2
        second = content[secondindex:secondindex_end]
        wrs[times[1]] = (float(second))
        #print(second)
        thirdindex = secondindex_end+3
        thirdindex_end = thirdindex + 5
        if content[thirdindex+2] != '.':
            thirdindex_end = thirdindex+2
        third = content[thirdindex:thirdindex_end]
        wrs[times[2]] = (float(third))

        fourthindex = thirdindex_end+3
        fourthindex_end = fourthindex + 5
        if content[fourthindex+2] != '.':
            fourthindex_end = fourthindex+2
        fourth = content[fourthindex:fourthindex_end]
        wrs[times[3]] = (float(fourth))

        fifthindex = fourthindex_end+3
        fifthindex_end = fifthindex + 5
        if content[fifthindex+2] != '.':
            fifthindex_end = fifthindex+2
        fifth = content[fifthindex:fifthindex_end]
        wrs[times[4]] = (float(fifth))

    #     fourth = content[index+38:index+43]
    #     wrs[times[3]] = (float(fourth))
    #     fifth = content[index+46:index+51]
    #     wrs[times[4]] = (float(fifth))
    return wrs

def getChampStats(champion,role):
    champ_url = "https://na.op.gg/champion/" + champion + "/statistics/" + role
    champ_page = requests.get(champ_url)
    champ_soup = BeautifulSoup(champ_page.content,'html.parser')
    winrateByLength = champ_soup.find(id="Trend-GameLengthWinRateGraph").next_siblings
    #print(champ_soup.encode('utf8'))
    #print()
    counter = 0
    times = ["0-25","25-30","30-35","35-40","40+"]
    #print("Champion: " + champion)
    winrates = {}
    for sibling in winrateByLength:
        if counter == 1:
            strsib = str(sibling.encode('utf8'))
            start_substr = "\"y\":"
            yindex = strsib.find(start_substr)
            matches = re.finditer(start_substr, strsib)
            matches_positions = [match.start() for match in matches]
            #print("Champion " + champion + ", Role: " + role)
            #print(matches_positions)
            for i in range(5):
                startindex = matches_positions[i]+4
                wr_num = ""
                temp = strsib[startindex]
                tempcount = 0
                while(temp != ',' and temp != '}' and tempcount <5):
                    wr_num= wr_num+(temp)
                    tempcount+=1
                    temp = strsib[startindex+tempcount]
                    
                winrates[times[i]] = float(str(wr_num))

        counter+=1
   
    return winrates

def getTeamPlayersData(roster,champions,roles, year, playerNames_oracle, team, is_opponent,isSpring):
    #size = 13 features
    stats_list_1 = ["Games","WR","KDA", "Gold/M","Dmg_per_gold"]
    stats_list_2=[ "Winrate","Gd/min","Avg_game_time", "FT_%","Drag_%","Gd_at_15"]
    stats_list_3 = ["0-25","25-30","30-35","35-40","40+"]
    O_stats_list_1 = ["O_Games","O_WR","O_KDA", "O_Gold/M","O_Dmg_per_gold"]
    O_stats_list_2=[ "O_Winrate","O_Gd/min","O_Avg_game_time", "O_FT_%","O_Drag_%","O_Gd_at_15"]
    O_stats_list_3 = ["O_0-25","O_25-30","O_30-35","O_35-40","O_40+"]
    teamDataDict = {}
    if(is_opponent == False):
        O_stats_list_1 = stats_list_1
        O_stats_list_2 = stats_list_2
        O_stats_list_3 = stats_list_3
    
    for i in range(len(stats_list_1)):
        teamDataDict[O_stats_list_1[i]] = 0
    for i in range(len(stats_list_2)):
        teamDataDict[O_stats_list_2[i]] = 0
    for i in range(len(stats_list_3)):
        teamDataDict[O_stats_list_3[i]] = 0
    

    for i in range(len(roster)):
        playerStats = getPlayerStats(roster[i],year,champions[i],playerNames_oracle[i],isSpring)
        print(playerStats)
        for j in range(len(stats_list_1)):
            # print(O_stats_list_1[j])
            # print(stats_list_1[j])
            #print(roster[i])
            print(champions[i])
            teamDataDict[O_stats_list_1[j]] += playerStats[stats_list_1[j]]
    for i in range(len(stats_list_1)):
        teamDataDict[O_stats_list_1[i]] /= 5
    
    if(team == "Counter Logic Gaming"):
        team = "CLG"
    if(team == "Team SoloMid"):
        team = "TSM"
    teamStats = getTeamData(team)
    for i in range(len(teamStats[team])):
        teamDataDict[O_stats_list_2[i]] = teamStats[team][i]
    champStats = [0,0,0,0,0]
    for i in range(len(champions)):
        curr_ChampStats = getChampStatsGG(champions[i])
        for j in range(len(champStats)):
            champStats[j] += curr_ChampStats[stats_list_3[j]]
    champStats[:] = [x / 5 for x in champStats ]
    for i in range(len(champStats)):
        teamDataDict[O_stats_list_3[i]] = champStats[i]
    return teamDataDict

#returns a dictionary of all features
def appendAllStats(roster1,roster2, year, champions1,champions2,roles,roster1_oracle,roster2_oracle,team1,team2,isSpring):
    is_opponent1 = False
    is_opponent2 = True
    first_team_dict = getTeamPlayersData(roster1,champions1,roles,year,roster1_oracle,team1,is_opponent1,isSpring)
    second_team_dict = getTeamPlayersData(roster2,champions2,roles,year,roster2_oracle,team2,is_opponent2,isSpring)  
    totalfeatures = {**first_team_dict, **second_team_dict}
    #print(first_team_dict)
    #print(second_team_dict)
    return totalfeatures
    # totalStats = pd.DataFrame()
    # totalPlayerStats = pd.DataFrame()
    # for i in range(len(players)):
    #     playerStats = getPlayerStats(players[i],year,champions[i],players_oracle[i])
    #     totalPlayerStats = totalPlayerStats.append(playerStats,ignore_index = True)
    # totalStats.append(totalPlayerStats,ignore_index=True)
    # totalChampStats = pd.DataFrame()
    # for i in range(len(champions)):
    #     champStats = getChampStats(champions[i],roles[i])
    #     totalChampStats= totalChampStats.append(champStats,ignore_index=True)
    # totalStats.append(totalChampStats,ignore_index=True)
    # frames = [totalPlayerStats, totalChampStats]

    # result = pd.concat(frames, axis = 1)
    # return result

def combineGames(filename):
    temp_team_list_1 = []
    temp_team_list_2 = []
    team1_list = []
    team2_list = []
    temp_champ_list_1 = []
    temp_champ_list_2 = []
    team1_champs = []
    team2_champs = []
    outcome_list = []
    team_list_1 = []
    team_list_2=[]
    temp_outcome_list = []
    first = True
    with open(filename, newline = '') as csvfile:
        readCSV = csv.reader(csvfile,delimiter = ',')
        first = True
        counter = 0
       # gamecounter = 0
     
        for row in readCSV:
        
            
            if first == True:
                first = False
                continue
            if(counter == 10):
                counter+=1
                team_list_1.append(row[14])
                
                temp_outcome_list.append(row[22])
            elif counter == 11:
                team1_list.append(temp_team_list_1)
                team2_list.append(temp_team_list_2)
                team1_champs.append(temp_champ_list_1)
                team2_champs.append(temp_champ_list_2)
                temp_outcome_list.append(row[22])
                team_list_2.append(row[14])
                outcome_list.append(temp_outcome_list)
                temp_team_list_1 = []
                temp_team_list_2 = []
                temp_champ_list_1 = []
                temp_champ_list_2 = []
                temp_outcome_list = []
                #gamecounter+=1
                counter = 0
            elif counter < 5:
                temp_team_list_1.append(row[13])
                temp_champ_list_1.append(row[15])
               
                counter+=1
            else:
                temp_team_list_2.append(row[13])
                temp_champ_list_2.append(row[15])
               
                counter+=1
            
           
    
    result = [team1_list,team2_list,team1_champs,team2_champs,outcome_list,team_list_1,team_list_2]
    return result

    # frames = []
    # for game in games:
    #     frames.append(pd.DataFrame.from_dict(game))
    # result = pd.concat(frames)
    # return result
#team_df = getTeamData("CLG")

def createTrainTestSets(gamePlayers1,gamePlayers2,gameChampions1,gameChampions2,gameOutcomes,teamList1,teamList2, year,roles,isSpring):
    output = pd.DataFrame()
    for i in range(len(gamePlayers1)):
        roster1 = gamePlayers1[i]
        roster2 = gamePlayers2[i]
        champions1 = gameChampions1[i]
        champions2 = gameChampions2[i]
        team1 = teamList1[i]
        team2=teamList2[i]
        example = appendAllStats(roster1,roster2,year, champions1,champions2,roles,roster1,roster2,team1,team2,isSpring)
        if gameOutcomes[i][0] == '1':
            example["Outcome"] = 0
        else:
            example["Outcome"] = 1
        #print(example)
        output = output.append(example,ignore_index=True)
    #print(output)
    return output


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label= 1), sum(target == y_pred) / float(len(y_pred))

def make_prediction(clf,features, team1, team2):
    y_pred = clf.predict(features)
    if y_pred == 1:
        print("Winner of " + team1 + " vs " + team2 + " = " + team2)
    else:
        print("Winner of " + team1 + " vs " + team2 + " = " + team1)
def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

def read_training_set(filename):
    df = pd.read_table(filename, delim_whitespace=True,header=0)
    return df
def read_training_labels(filename):
    df = pd.read_table(filename, delim_whitespace=True,header=0) 
    return df
def featurescaling(example,filename,X_all):
    colnames = []
    for col in X_all.columns:
        colnames.append(col) 
    means = {}
    stds = {}
    unscaled = read_training_set(filename)
    for name in colnames:
        mean = sum(unscaled[name])/len(unscaled[name])
        means[name] = mean
        variance = sum([((x - mean) ** 2) for x in unscaled[name]]) / len(unscaled[name]) 
        res = variance ** 0.5
        stds[name] = res

    for ind in example.index:
        for name in colnames:
            temp = example[name][ind] 
            #print(temp)
            tempmean = means[name]
            
            tempstd = stds[name]
            
            example[name][ind] = (temp - tempmean)/tempstd
    return [means,stds]
def gamePrediction(X_all,players1,players2,champions1,champions2,roles,team1,team2):
    example2 = appendAllStats(players1,players2,2020, champions1,champions2,roles,players1,players2,team1,team2,False)
    example2 = pd.DataFrame(example2,index=[0])
    featurescaling(example2,"unscaledvalues2.txt",X_all)
    make_prediction(clf_A,example2,team1,team2)

def alreadytrained(clf_A,X_all):

    
    teams_list = ["100 Thieves", "CLG", "Cloud9", "Dignitas","Evil Geniuses", "FlyQuest", "Golden Guardians", "Immortals", "Team Liquid", "TSM"]
    clg = ["Ruin","Wiggily","Pobelter","Stixxay","Smoothie"]
    dig = ["V1per","Dardoch","Fenix","Johnsun","Aphromoo"]
    tl = ["Impact","Broxah","Jensen","Tactical","CoreJJ"]
    cloud9 = ["Licorice","Blaber","Nisqy","Zven","Vulcan%20(Philippe%20Laflamme)"]
    eg = ["Huni","Svenskeren","Goldenglue","Bang","Zeyzal"]
    imt = ["Allorim","Xmithie","Insanity","Apollo","Hakuho"]
    cloud9_oracle = ["Licorice","Blaber","Nisqy","Zven","Vulcan"]
    players_oracle = ["Licorice","Blaber","Nisqy","Zven","Vulcan", "Ruin","Wiggily","Pobelter","Stixxay","Smoothie"]
    champions = ["Wukong","Lee Sin","Rumble","Ezreal","Rakan","Malphite","Trundle","Syndra","Aphelios","Nautilus"]
    dig_champs = ["Volibear","Graves","Orianna","Ashe","Blitzcrank"]
    flyq = ["Solo","Santorin","PowerOfEvil","WildTurtle","igNar"]

    # c9_champs = ["Shen","Hecarim","Sett","Sona","Lux"]
    # tl_champs = ["Mordekaiser","Trundle","Syndra","Ezreal","Bard"]
    

    #badfeatures = ["0-25","25-30","30-35","35-40","40+","O_0-25","O_25-30","O_30-35","O_35-40","O_40+"] 

    # example1 = appendAllStats(cloud9,tl,2020, c9_champs,tl_champs,roles,cloud9,tl,"Cloud9","Team Liquid",False)
    # example1 = pd.DataFrame(example1,index=[0])

    # featurescaling(example1,"unscaledvalues2.txt",X_all)

    # make_prediction(clf_A,example1,"c9","tl")
    # print(example1)

    # tl_champs2 = ["Kennen","Volibear","Zoe","Ezreal","Bard"]
    # flyq_champs = ["Renekton","Sett","Orianna","Ashe","Rakan"]
    # example2 = appendAllStats(tl,flyq,2020, tl_champs2,flyq_champs,roles,tl,flyq,"Team Liquid","FlyQuest",False)
    # example2 = pd.DataFrame(example2,index=[0])
    # featurescaling(example2,"unscaledvalues2.txt",X_all)

    # make_prediction(clf_A,example2,"tl","flyq")

    # eg_champs = ["Gangplank","Volibear","Zoe","Aphelios","Alistar"]
    # imt_champs = ["Urgot","Graves","Azir","Ashe","Braum"]
    # example3 = appendAllStats(eg,imt,2020, eg_champs,imt_champs,roles,eg,imt,"Evil Geniuses","Immortals",False)
    # example3 = pd.DataFrame(example3,index=[0])
    # featurescaling(example3,"unscaledvalues2.txt",X_all)

    # make_prediction(clf_A,example3,"eg","imt")

    hundredt = ["Ssumday","Contractz","Ryoma","Cody Sun","Poome"]
    # hundredt_champs = ["Wukong","Graves","Galio","Ashe","Thresh"]
    # clg_champs = ["Kennen","Sett","Orianna","Xayah","Bard"]
    # example4 = appendAllStats(hundredt,clg,2020, hundredt_champs,clg_champs,roles,hundredt,clg,"100 Thieves","CLG",False)
    # example4 = pd.DataFrame(example4,index=[0])
    # featurescaling(example4,"unscaledvalues2.txt",X_all)
    # make_prediction(clf_A,example4,"100t","clg")

    tsm = ["Broken Blade","Spica","Bjergsen","Doublelift","Treatz"]
    # tsm_champs = ["Wukong","Sett","Syndra","Xayah","Rakan"]
    # dig_champs = ["Jayce","Karthus","Ekko","Ashe","Tahm Kench"]
    # example5 = appendAllStats(tsm,dig,2020, tsm_champs,dig_champs,roles,tsm,dig,"TSM","Dignitas",False)
    # example5 = pd.DataFrame(example5,index=[0])
    
    # featurescaling(example5,"unscaledvalues2.txt",X_all)
    # print(example5)
    # make_prediction(clf_A,example5,"tsm","dig")

    gg = ["Hauntzer","Closer","Damonte","FBI","huhi"]
    # gg_champs = ["Kennen","Volibear","Galio","Ezreal","Nautilus"]
    # flyq_champs2 = ["Ornn","Sett","Azir","Aphelios","Rakan"]
    # example6 = appendAllStats(gg,flyq,2020, gg_champs,flyq_champs2,roles,gg,flyq,"Golden Guardians","FlyQuest",False)
    # example6 = pd.DataFrame(example6,index=[0])
    # featurescaling(example6,"unscaledvalues2.txt",X_all)
    # make_prediction(clf_A,example6,"gg","flyq")
    
    
    # imt_champs2 = ["Maokai","Sett","Zoe","Ashe","Nautilus"]
    # c9_champs2 = ["Kennen","Volibear","Galio","Kalista","Morgana"]
    # example7 = appendAllStats(imt,cloud9,2020, imt_champs2,c9_champs2,roles,imt,cloud9,"Immortals","Cloud9",False)
    # example7 = pd.DataFrame(example7,index=[0])
    # featurescaling(example7,"unscaledvalues2.txt",X_all)
    # make_prediction(clf_A,example7,"imt","c9")

    # dig_champs2 = ["Volibear","Graves","Orianna","Ashe","Blitzcrank"]
    # eg_champs2 = ["Camille","Sett","Galio","Aphelios","Thresh"]
    # example8 = appendAllStats(dig,eg,2020, dig_champs2,eg_champs2,roles,dig,eg,"Dignitas","Evil Geniuses",False)
    # example8 = pd.DataFrame(example8,index=[0])
    # featurescaling(example8,"unscaledvalues2.txt",X_all)
    # make_prediction(clf_A,example8,"dig","eg")

    # clg_champs2 = ["Gnar","Graves","Galio","Ashe","Braum"]
    # tsm_champs2 = ["Ornn","Jarvan IV","Cassiopeia","Ezreal","Karma"]
    # example9 = appendAllStats(clg,tsm,2020, clg_champs2,tsm_champs2,roles,clg,tsm,"CLG","TSM",False)
    # example9 = pd.DataFrame(example9,index=[0])
    # featurescaling(example9,"unscaledvalues2.txt",X_all)
    # make_prediction(clf_A,example9,"clg","tsm")

    # hundredt_champs2 = ["Kennen","Lee Sin","Orianna", "Ashe", "Bard"]
    # gg_champs2 = ["Karma","Olaf","Zoe","Aphelios","Sett"]
    # gamePrediction(X_all,hundredt,gg,hundredt_champs2,gg_champs2,roles,"100 Thieves","Golden Guardians")
    eg_champs2 = ["Sett","Trundle","Zoe", "Aphelios", "Nautilus"]
    tsm_champs2 = ["Maokai","Volibear","Syndra","Kalista","Thresh"]
    gamePrediction(X_all,cloud9,gg,eg_champs2,tsm_champs2,roles,"Cloud9","Golden Guardians")
def retrain(year,roles,filename,isSpring):
    
    result = combineGames(filename)

    gamePlayers1 = result[0]
    gamePlayers2 = result[1]
    gameChampions1 = result[2]
    gameChampions2 = result[3]
    gameOutcomes = result[4]
    teamList1 = result[5]
    teamList2 = result[6]
    output = createTrainTestSets(gamePlayers1,gamePlayers2,gameChampions1,gameChampions2,gameOutcomes,teamList1,teamList2,year,roles,isSpring)
    colnames = []



    X_all = output.drop(["Outcome"],1)
    for col in X_all.columns:
        colnames.append(col)
    y_all = output["Outcome"]
    print("Y Values")
    print(y_all)
    print()
    print("X Values")
    print(X_all)
    print()
    print("X values scaled")
    for col in colnames:
        X_all[col] = scale(X_all[col])
    print(X_all)
    return[X_all,y_all]
#team = getTeamPlayersData(clg, clg_champs, roles, 2020, clg, "CLG")
#print(team)    e
#totalStats = appendAllStats(clg,cloud9, 2020,clg_champs,c9_champs,roles, clg,cloud9_oracle,"CLG","Cloud9")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Uncomment if you want to get new training set 
roles = ["Top", "Jungle","Middle","adc","Support"]
#[X_all,y_all] = retrain(2020,roles)
#[X_train,y_train] = retrain(2020,roles,"springdata.csv",True)
#uncomment if you have already trained data
# filename = "x_input.txt"
# y_filename = "y_input.txt"
# X_all = read_training_set(filename)
# y_all = read_training_labels(y_filename)
# X_all = X_all.astype(float)
# y_all = y_all.astype(float)
# y_all = y_all["Y_Values"]
filename = "trainedxdata.txt"
y_filename = "trainedydata.txt"
X_train = read_training_set(filename)
y_train = read_training_labels(y_filename)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
y_train = y_train["Y_Values"]
testfilename = "testinputs.txt"
testoutfilename = "testout.txt"
#[X_test,y_test] = retrain(2020,roles,"summerdata.csv",False)
X_test = read_training_set(testfilename)
y_test= read_training_labels(testoutfilename)
X_test = X_test.astype(float)
y_test = y_test.astype(float)
y_test = y_test["Y_Values"]
#X_train,X_test,y_train,y_test = train_test_split(X_all,y_all,test_size=74,random_state=5,stratify=y_all)

clf_A = LogisticRegression(random_state = 42)
#getChampStatsGG("Elise")
train_predict(clf_A,X_train,y_train,X_test,y_test)
alreadytrained(clf_A,X_train)


# filename = "x_input.txt"
# y_filename = "y_input.txt"

#TODO
#add derivative of last 5 games to obtain a teams "trend"
#if player has never play4ed champ before, take overall stats of champ in pro play combined with the players overall stats
#see how the model is affected if we just take the difference of the equivalent features for team 1 and team 2





#totalStats = pd.DataFrame(totalStats, index=[0])
#print(totalStats)
#totalStatsNumerical = totalStats.drop(["Player","Champion"], axis = 1)
#print(result)
#output.drop(output.columns.difference(["0-25","25-30","30-35","35-40","40+","O_0-25","O_25-30","O_30-35","O_35-40","O_40+","Outcome"]), 1, inplace=True)
# print(output)
# scatter_matrix(output, diagonal = 'kde')
# plt.show()
# pd.set_option("display.max_rows", None, "display.max_columns", None) # more options can be specified also
# #print(totalStats)
#getChampStats("Ashe","adc")
#getPlayerStats("Licorice",2020,"Sett")
#print(totalPlayerStats)


# import mwclient
# import time
# import datetime as dt
# from datetime import date, timedelta
# site = mwclient.Site('lol.gamepedia.com', path='/')

# response = site.api('RunQuery',
# 	limit = 'max',
# 	tables = "ScoreboardGames=SG, ScoreboardPlayers=SP",
#     join_on = "SG.UniqueGame=SP.UniqueGame",
# 	fields = "SG.Tournament, SG.DateTime_UTC, SG.Team1, SG.Team2, SG.Winner, SG.Patch, SP.Link, SP.Team, SP.Champion, SP.SummonerSpells, SP.KeystoneMastery, SP.KeystoneRune, SP.Role, SP.UniqueGame, SP.Side",
# 	where = "SG.DateTime_UTC >= '2020-01-01 00:00:00'" #Results after Aug 1, 2019
# )
# print(response)