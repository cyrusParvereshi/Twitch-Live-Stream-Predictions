

# Example: sample input as a dict (keys must match your features)
def predict_viewers(duration_hrs=10, lang="en", game="Counter Strike", is_mature=True):
    sample_dict = {
        "duration_min": 60 * duration_hrs, #convert hours to minutes
        "language": lang,
        "game_name": game,
        "is_mature": is_mature
    }
    
    sample_df = pd.DataFrame([sample_dict])
    # 2. Preprocess sample with pipeline
    sample_pre = full_pipeline.transform(sample_df)
    # 3. Predict with your Keras model
    pred_log = model1.predict(sample_pre).flatten()[0]
    pred = np.expm1(pred_log)  # If you trained on log1p(y)
    print("Predicted Number of Viewers:", pred)
    return pred

def other_preds():
	viewer_preds_stream_time = [predict_viewers(x) for x in range(1, 24)]
	plt.plot(viewer_preds_stream_time)
	#X_train['language'].value_counts()
	languages = ['en', 'es', 'ru', 'de', 'pt', 'fr']
	viewer_preds_langs = [predict_viewers(lang=x) for x in languages]
	plt.plot(viewer_preds_langs)
	plt.xticks(list(range(len(languages))), languages) # Set tick locations and labels
	plt.show()
	predict_viewers(is_mature=False)
	games = ['Minecraft','ROBLOX', 'League of Legends', 'Madden NFL 25', 'Genshin Impact', 'Street Fighter 6', 'Mario Kart World']
	viewer_preds_games = [predict_viewers(game=x) for x in games]
	plt.plot(viewer_preds_games)
	plt.xticks([0,1,2,3,4,5,6], viewer_preds_games) # Set tick locations and labels
	plt.show()
	list(range(len(viewer_preds_games)))

if __name__ == '__main__':
	predict_viewers()