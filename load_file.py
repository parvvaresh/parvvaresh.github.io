import pandas as pd
class File:
	def __init__(self, path_sad, path_joy):
		self.path_sad = path_sad
		self.path_joy = path_joy

	def fit(self):
		temp = pd.read_csv(self.path_sad)
		temp = temp.iloc[0 : 500]
		df_sad = pd.DataFrame()
		df_sad["Text"] = temp["tweet"]
		df_sad["emotin"] = [0 for _ in range(df_sad.shape[0])]

		temp = pd.read_csv(self.path_joy)
		temp = temp.iloc[0 : 500]
		df_joy = pd.DataFrame()
		df_joy["Text"] = temp["tweet"]
		df_joy["emotin"] = [1 for _ in range(df_joy.shape[0])]
		df = pd.concat([df_sad, df_joy], axis=0)

		return df

