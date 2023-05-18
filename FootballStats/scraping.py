import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import requests
import time

pd.set_option("display.width", 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

standings_url = "https://fbref.com/en/comps/26/Super-Lig-Stats"

data = requests.get(standings_url)
data_ = requests.get(standings_url)


soup = BeautifulSoup(data_.text)
standings_table = soup.select('table.stats_table')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]

team_urls = [f"https://fbref.com{l}" for l in links]

data = requests.get(team_urls[0])


matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]

data = requests.get(f"https://fbref.com{links[0]}")

shooting = pd.read_html(data.text, match="Shooting")[0]

shooting.head()

shooting.columns = shooting.columns.droplevel()

team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "PK", "PKatt"]], on="Date")
team_data.head()


years = list(range(2023, 2021, -1))

all_matches = []

for year in years:
    standings_url = "https://fbref.com/en/comps/26/Super-Lig-Stats"

    while True:
        data = requests.get(standings_url)
        soup = BeautifulSoup(data.text, "html.parser")
        standings_table = soup.select('table.stats_table')[0]

        links = [l.get("href") for l in standings_table.find_all('a')]
        links = [l for l in links if '/squads/' in l]
        team_urls = [f"https://fbref.com{l}" for l in links]

        previous_season = soup.select("a.prev")[0].get("href")
        standings_url = f"https://fbref.com{previous_season}"

        for team_url in team_urls:
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
            data = requests.get(team_url)
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
            soup = BeautifulSoup(data.text, "html.parser")
            links = [l.get("href") for l in soup.find_all('a')]
            links = [l for l in links if l and 'all_comps/shooting/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            shooting = pd.read_html(data.text, match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()
            try:
                team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "PK", "PKatt"]], on="Date")
            except ValueError:
                continue
            team_data = team_data[team_data["Comp"] == "Süper Lig"]

            team_data["Season"] = year
            team_data["Team"] = team_name
            all_matches.append(team_data)
            time.sleep(1)

        if previous_season is None:
            break

len(all_matches)

match_df = pd.concat(all_matches)

match_df.columns = [c.lower() for c in match_df.columns]

match_df

match_df.to_csv("matches.csv")

