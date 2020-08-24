# PUBG-analysis-and-modeling
## Table of Contents
<ul>
<li><a href="#Dictionary">Data Dictionary</a></li>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
<li><a href="#model">Modeling </a></li>
</ul> 

<a id='Dictionary'></a>
## Data Dictionary
- `you can find the data here ---> https://www.kaggle.com/c/pubg-finish-placement-prediction <---
- **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
- **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.
- **assists** - Number of enemy players this player damaged that were killed by teammates.
- **boosts** - Number of boost items used.
- **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.
- **DBNOs** - Number of enemy players knocked.
- **headshotKills** - Number of enemy players killed with headshots.
- **heals** - Number of healing items used.
- **killPlace** - Ranking in match of number of enemy players killed.
- **killPoints** - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.)
- **kills** - Number of enemy players killed.
- **killStreaks** - Max number of enemy players killed in a short amount of time.
- **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a - player and driving away may lead to a large longestKill stat.
- **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
- **numGroups** - Number of groups we have data for in the match.
- **revives** - Number of times this player revived teammates.
- **rideDistance** - Total distance traveled in vehicles measured in meters.
- **roadKills** - Number of kills while in a vehicle.
- **swimDistance** - Total distance traveled by swimming measured in meters.
- **teamKills** - Number of times this player killed a teammate.
- **vehicleDestroys** - Number of vehicles destroyed.
- **walkDistance** - Total distance traveled on foot measured in meters.
- **weaponsAcquired** - Number of weapons picked up.
- **winPoints** - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.)
- **winPlacePerc** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

<a id='intro'></a>
## Introduction

- In this project what we want is analyzing the data and find out what affects the player's win by answering some questions such as:
    -  **Who are the highest win (Solos, Duos or Squads)?**
    -  **What is the impact of damage on the number of kills?**
    -  **What is affects of the player's win?**
    -  **What is the amount of work as a team between players?**


- **What's the best strategy to win in PUBG? Should you sit in one spot and hide your way into victory, or do you need to be the top shot? Let's let the data do the talking!**


 <a id='conclusions'></a>
# Conclusions

**After analyzing and understanding the data is time to answer a question:**
# What's the best strategy to win in PUBG?

### Whenever you played in a team increased your chance to win and find aid to have other opportunities, In addition to increasing the number of killers you kill them , Taking into account that if the competitor kills more than you, you are an exhibition of loss so you must develop your skills in murder from distances and focus when shooting on the head, Must move so much until they do not give an opportunity to hit you and always try to collect weapons and find vehicles
