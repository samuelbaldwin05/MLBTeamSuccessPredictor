"""
DS2500
Spring 2024
Final Project 

Jack Carroll, Alexander Tu, Samuel Baldwin
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import statistics
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
import seaborn as sns
import math
from sklearn.metrics import accuracy_score

DIRNAME = "data" 
TEAMS = ["CIN", "DET", "BOS", "ATL", "PHI", "PIT"]

def get_filenames(dirname, ext = ".csv"):
    """ given a directory name, return the full path and name for every
        non directory file in the directory as a list of strings
    """
    filenames = []
    files = os.listdir(dirname)
    for file in files:
        if not os.path.isdir(file) and file.endswith(ext):
            filenames.append(dirname + "/" + file)
    return filenames

def create_dct(files):
    """ Given a list of filenames, this function creates a dictionary where 
        each key in the name of the file minus extensions, and the value is 
        a dataframe corresponding to the file
    """
    dct = {}
    for file in files:
        dct[file[5:][:-4]] = pd.read_csv(file)
    return dct

def filter_year(df, series, year):
    """ Given a datfame with a series representing year and a year to work
        with, this function deletes all rows of the dataframe before the given
        year
    """
    if series in df.columns:
        df = df[(df[series] == year).idxmax():]
    return df        

def get_stat(df, year, series, team = 0):
    """ given a dataframe, a string for a team name(optional), a year, and a 
        string for a series, this function will find the statistics in the 
        given series of the given dataframe from the year, and only from the 
        team if a teamid is given
    """
    if team == 0:
        temp = df[df["yearid"] == year]
        return temp[series]
    else:
        temp = df[df["yearid"] == year]
        temp = temp[temp["teamid"] == team]
        return temp[series]

def find_corrs(df, teams):
    """ Given a df for statistics, a df for win statistics, and an index to 
        slice certain columns at, this function calculates the correlation of 
        the average of all variables in the df for a team for each year from 
        1972-2014, to the winning percentages of the team for that year
    """
    dct = {}
    for ser in list(df)[14:39]:
        corrs = []
        for team in teams:
            stats = []
            wins = []
            for y in range(1972, 2015):
                w = list(get_stat(df, y, "w", team))
                l = list(get_stat(df, y, "l", team))
                wins.append(sum(w))
                s = get_stat(df, y, ser, team)
                s = s.fillna(0)
                stats.append(statistics.mean(s))
            corrs.append(statistics.correlation(wins, stats))
        dct[ser] = statistics.mean(corrs)
    dct = dict(sorted(dct.items(), key=lambda item: item[1], reverse = True))
    stat_names = dct.keys()
    corrs = dct.values()
    return pd.DataFrame(data = {"Correlations": corrs}, index = stat_names)

def plot_corrs(df, threshold):
    """ given a dataframe containing only correlation data, this function will
        plot the correlations above the given threshold
    """
    df2 = df[abs(df["Correlations"]) >= threshold]
    df2.plot(kind = "bar", color = "maroon", 
              title = "Correlation of Statistics to Team Wins")
    plt.xlabel("Statistic")
    plt.ylabel("Correlation")
    
def multiple_regression(df, features, targ, year):
    '''
    Create a multiple regression given features and a targ variable from
    a given dataframe
    Additionally, given a future year predict the target variable
    given the feature variable from that year using the linear
    regression model
    Return the predicted dataframe
    '''
    df2 = df.copy()
    coef_features = features.copy()
    features.append(targ)
    df = df[features]

    X_train, X_test, y_train, y_test = train_test_split(df[coef_features],
                                    df[targ], random_state=0, test_size= .3)
    lr = LinearRegression()
    lr.fit(X = X_train, y = y_train)
    
    features.append("yearid")
    features.append("teamid")
    filtered_df = df2[features]
    filtered_df = filtered_df[filtered_df['yearid'] == year]
    year_predicted = lr.predict(filtered_df[coef_features])
    
    return year_predicted

def create_histo(actual, predicted, title, x_label, y_label, bins):
    '''
    Create a histogram that compares actual and predicted values
    for a train test prediction 
    Rreturn nothing
    '''
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.hist((actual - predicted),
            bins, range = [-10,10], align='mid')
    plt.show()
    
def bar_compare(df, title, x_label, y1, y2, bins):
    '''
    Create a bar graph that compares two data points on the same axis
    Return nothing
    '''
    df.head(bins).plot(x = x_label, y = [y1, y2], kind = "bar")  
    plt.title(title)
    plt.show()
    
def data_metrics(df1, df2):
    '''
    Given two dataframes, calculate the mean absolute error,
    mean square error, and root mean squared error between the two
    Print and return the results
    '''
    meanAbsErr = metrics.mean_absolute_error(df1, df2)                               
    meanSqrErr = metrics.mean_squared_error(df1, df2)
    squarerootMeanSqrErr = np.sqrt(meanSqrErr)
    
    return meanAbsErr, meanSqrErr, squarerootMeanSqrErr

def knn_teams(df, feature_one, feature_two):
    """ Given the dataframe on teams, it will take in two features to use for a
        KNN algorithm that will predict if a team made the playoffs and how far
        into the playoffs they went. This function also creates two vizualizations
        for the results of the knn classification in order to make it clearer
        for the user what happened
    """
    data = df[[feature_one, feature_two]]
    labels = df["playoff_label"]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 0)

    # Initialize model
    knn = KNN(n_neighbors= int(round(math.sqrt(len(X_train)), 0)))
    
    # Train model
    knn.fit(X_train, y_train)
    
    # Generate Predictions
    predictions = knn.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Create Value Map
    value_dict = {True: 'Correct', False: 'Incorrect'}
    
    # Clean Data
    predictions = pd.Series(predictions)
    y_test = y_test.reset_index()["playoff_label"]
    predictions.name = "playoff_label"
    
    # Create a dataframe with all of the information we will need for our two vizualizations
    viz_df = X_test.reset_index()[[feature_one, feature_two]]
    viz_df["predictions"] = (y_test == predictions).map(value_dict)
    viz_df["playoff_label"] = y_test
    
    
    # Create vizualization
    sns.set_palette(sns.color_palette("Paired"))
    accuracy_plot = sns.scatterplot(viz_df, x=feature_one, y=feature_two, hue="predictions")
    accuracy_plot.set(title = 'Model Performance: Where Did Our Model Miss?')
    plt.show()
    
    # Vizual aspects
    sns.set_palette(sns.color_palette('bright'))
    viz_df["playoff_label"] = pd.Series(y_test)
    sns.scatterplot(viz_df, x=feature_one, y=feature_two, hue="playoff_label").set(title = 'Actual Test Values')
    plt.show()

    print(viz_df)
    print(f'The accuracy of our model is {accuracy:.2f}')

def create_playoff_label(teams):
    """ Given the 'teams' dataframe, creates a new column called 'playoff_label' where each
        case is given a number to represent whether a team did not make the playoffs
        all the way to winning the entire playoffs
    """
    new_col = []
    
    # creating the 'playoff_label' column
    for i in range(len(teams)):
        # A team won the world series
        if (teams.loc[i, "WSWin"] == "Y"):
            new_col.append(4)
        # A team only won their league in the playoffs
        elif (teams.loc[i, "LgWin"] == "Y"):
            new_col.append(3)   
        # A team won their division in the regular season for a playoff spot
        elif (teams.loc[i, "DivWin"] == "Y"):
            new_col.append(2)
        # A team won their Wild Card game for a playoff spot
        elif (teams.loc[i, "WCWin"] == "Y"):
            new_col.append(1)
        # A team did not make the playoffs at all
        else:
            new_col.append(0)
    
    # adding the new column to the dataframe
    teams["playoff_label"] = new_col
    return teams

def create_playoff_label_simple(teams):
    """ Given the 'teams' dataframe, creates a new column called 
        'playoff_label' where each case is a 0, which means the team did not 
        make the playoffs, or a 1, which means the team did make the playoffs
    """
    new_col = []
    
    for i in range(len(teams)):
        # A team made the playoffs
        if (teams.loc[i, "wswin"] == True) or (teams.loc[i, "wcwin"] == True):
            new_col.append(1)
        # A team did not make the playoffs at all
        else:
            new_col.append(0)

    # adding the new column to the dataframe
    teams["playoff_label"] = new_col
    return teams


    
def main():
    files = get_filenames(DIRNAME)
    data = create_dct(files)

    for d in data:
        data[d] = filter_year(data[d], "yearid", 1972)
    
    correlations = find_corrs(data["teams"], TEAMS)
    plot_corrs(correlations, 0.4)

    # Create a dataframe predicting the number of wins a team will have 
    # in 2015 from their era, runs, and at bat statistics
    pred_2015 = pd.DataFrame(multiple_regression(data["teams"],
                                ["era", "r", "h", "sv"], "w", 2015))
    pred_2015.columns = ["p"]
    
    # Create a dataframe with the team name and actual wins
    left = (get_stat(data["teams"], 2015, "teamid"))
    right = (get_stat(data["teams"], 2015, "w"))
    wins_df = pd.merge(left, right, left_index=True, right_index=True)

    # Weight the predicted dataframe to account for the sum of the 
    # predicted wins being different than the total number of possible
    # wins in a season
    pred_weighted = (pred_2015["p"] * (wins_df['w'].sum()/
                                     pred_2015['p'].sum())).astype(int)

    # Combine the dataframes of predicted and actual and sort by 
    # actual wins to make readable, clean data
    pred_weighted.index = wins_df.index
    result_df = pd.concat([wins_df, pred_weighted], axis=1)
    result_df.columns = ['Team', 'Wins', 'Predicted Wins']
    sorted_df = result_df.sort_values(by=['Wins'], ascending = [False])
    #print(sorted_df)
    
    
    # Create a bar graph with two bars for a random selection of 
    # 10 teams, showing predicted wins and actual wins
    bar_compare(result_df, "Actual Wins vs Predicted Wins of 2015 MLB Teams",
                "Team", "Wins", "Predicted Wins", 10)
    
    # Create a histogram that shows the actual wins minus the predicted 
    # wins to show how accurate the regression was on average
    create_histo(result_df["Wins"], result_df["Predicted Wins"], 
                 "MLB 2015: Difference in Actual vs. Predicted Wins", 
                 "Actual wins - Predicted wins", "Number of Teams", 9)
    
    # Calculate the MAE, MSE, RMSE
    metrics = data_metrics(result_df["Wins"], result_df["Predicted Wins"])
    #print('Mean Absolute Error:', metrics[0])
    #print('Mean Square Error:', metrics[1])
    #print('Root Mean Square Error:', metrics[2])
    
    team_data = data["teams"].reset_index()
    #print(team_data)
    #print(team_data.loc[0, "divwin"])
    new_teams = create_playoff_label_simple(team_data)
    

    print(new_teams.head())
    knn_teams(new_teams, "era", "r")
    
    zero_count = 0
    for i in range(len(new_teams)):
        if new_teams.loc[i, "playoff_label"] == 0:
            zero_count += 1
    
    zero_ratio = round((zero_count / len(new_teams["playoff_label"])), 2)
    print("The percent of zeros to the total amount of data points is",
          zero_ratio)
    

if __name__ == "__main__":
    main()
