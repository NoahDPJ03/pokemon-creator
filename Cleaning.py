import pandas as pd

df = pd.read_csv("pokemon_data.csv")


stats_list = pd.json_normalize(df['stats'].apply(eval))
df = pd.concat([df.drop(columns=['stats']), stats_list], axis=1)

df = pd.concat([df.drop(columns=['types']), df['types'].apply(pd.Series)], axis=1)
df = df.rename(columns={0: "type_list"})
df = pd.concat([df.drop(columns=['abilities']), df['abilities'].apply(pd.Series)], axis=1)
df = df.rename(columns={0: "ability_list"})

df.drop(columns=["id"], inplace=True)

df.to_csv("pokemon_data_cleaned.csv", index=False)