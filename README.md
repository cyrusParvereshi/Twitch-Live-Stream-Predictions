Explanation of files: 

1. crawl_twitch.py -> collects the current livestreams on twitch
2. merge_runs.py -> combine all collected data into 1 giant csv
3. data_cleaning.py -> cleans all bad rows/cols, preprocesses data for models
4. feature_selection.py -> performs feature selection for linear regression
5. final.ipynb -> contains the MLR, RandomForestRegressor, and DNN model training and predictions
6. citations.txt -> sources cited in IEEE format for this project
7. creds.ini needs to be created with twitch app credentials if want to test project data collection