"""This module create the data used for model training and testing."""

import pandas as pd
import numpy as np

game_data = pd.read_csv("data/input/nba.games.stats.csv")


def get_season(str_season):
    """Get Season"""
    str_year = str_season[:4]
    str_month = str_season[5:7]
    if str_month in ("01", "02", "03", "04", "05", "06"):
        return int(str_year) - 1
    if str_month in ("07", "08", "09", "10", "11", "12"):
        return int(str_year)
    return None


game_data["Season"] = game_data["Date"].apply(get_season)


def get_division(team):
    """Get Division"""
    if team in ("TOR", "PHI", "BOS", "BRK", "NYK"):
        division = "Atlantic"
    elif team in ("MIL", "IND", "DET", "CHI", "CLE"):
        division = "Central"
    elif team in ("ORL", "CHO", "MIA", "WAS", "ATL"):
        division = "Southeast"
    elif team in ("DEN", "POR", "UTA", "OKC", "MIN"):
        division = "Northwest"
    elif team in ("GSW", "LAC", "SAC", "LAL", "PHO"):
        division = "Pacific"
    elif team in ("HOU", "SAS", "MEM", "NOP", "DAL"):
        division = "Southwest"
    else:
        division = None
    return division


game_data["Division"] = game_data["Team"].apply(get_division)


def get_conference(division):
    """Get Conference"""
    if division in ("Atlantic", "Central", "Southeast"):
        return "East"
    if division in ("Northwest", "Pacific", "Southwest"):
        return "West"
    return None


game_data["Conference"] = game_data["Division"].apply(get_conference)
game_data["inWestConference"] = (game_data["Conference"] == "West")

game_data["atHome"] = (game_data["Home"] == "Home")

game_data["GameDate"] = pd.to_datetime(game_data["Date"], format="%Y-%m-%d")
game_data["LastGameDate"] = game_data.groupby(["Season", "Team"])["GameDate"].shift(
    1, fill_value=np.nan
)
game_data["NumDaysRested"] = np.where(
    game_data["Game"] == 1,
    np.nan,
    (game_data["GameDate"] - game_data["LastGameDate"]).dt.days,
)
game_data["RestedOneDay"] = np.where(
    game_data["Game"] == 1, np.nan, (game_data["NumDaysRested"] == 1)
)
game_data["RestedTwoDays"] = np.where(
    game_data["Game"] == 1, np.nan, (game_data["NumDaysRested"] == 2)
)

# Create Win Percentage Data
game_data["WinsAdded"] = np.where(game_data["WINorLOSS"] == "W", 1, 0)
game_data["CumSumWinsAfterGame"] = game_data.groupby(["Season", "Team"])[
    "WinsAdded"
].cumsum()
game_data["CumSumWinsBeforeGame"] = game_data.groupby(["Season", "Team"])[
    "CumSumWinsAfterGame"
].shift(1, fill_value=np.nan)
game_data["WinPctBeforeGame"] = np.where(
    game_data["Game"] > 1,
    game_data["CumSumWinsBeforeGame"] / (game_data["Game"] - 1),
    np.nan,
)
game_data = game_data.drop(
    columns=["WinsAdded", "CumSumWinsAfterGame", "CumSumWinsBeforeGame"]
)

# Create Average Point Differential Data
game_data["PointDiff"] = game_data["TeamPoints"] - game_data["OpponentPoints"]
game_data["CumSumPointDiffAfterGame"] = game_data.groupby(["Season", "Team"])[
    "PointDiff"
].cumsum()
game_data["CumSumPointDiffBeforeGame"] = game_data.groupby(["Season", "Team"])[
    "CumSumPointDiffAfterGame"
].shift(1, fill_value=np.nan)
game_data["AvgPointDiffBeforeGame"] = np.where(
    game_data["Game"] > 1,
    game_data["CumSumPointDiffBeforeGame"] / (game_data["Game"] - 1),
    0,
)
game_data = game_data.drop(
    columns=["CumSumPointDiffAfterGame", "CumSumPointDiffBeforeGame"]
)

game_data = game_data.merge(
    game_data[
        [
            "Date",
            "Opponent",
            "Team",
            "WinPctBeforeGame",
            "AvgPointDiffBeforeGame",
            "NumDaysRested",
        ]
    ],
    left_on=["Date", "Team", "Opponent"],
    right_on=["Date", "Opponent", "Team"],
    suffixes=("", "_Opponent"),
)
game_data = game_data.drop(columns=["Opponent_Opponent", "Team_Opponent"])

game_data["AvgPointDiffBeforeGameMinusOpponent"] = (
    game_data["AvgPointDiffBeforeGame"] - game_data["AvgPointDiffBeforeGame_Opponent"]
)

game_data["WinPctBeforeGameMinusOpponent"] = (
    game_data["WinPctBeforeGame"] - game_data["WinPctBeforeGame_Opponent"]
)

game_data["NumDaysRestedMinusOpponent"] = (
    game_data["NumDaysRested"] - game_data["NumDaysRested_Opponent"]
)

game_data.to_csv("data/intermediate/dataForPrediction.csv")
