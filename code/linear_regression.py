import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def read_data():
    df = pd.read_csv('../data/pakistan_temps.csv')
    df["date"] = pd.to_datetime(
        df[" Year"].astype(str) + " " + df["Month"] + " 15",
        format="%Y %B %d"
    )
    df.index = df["date"]
    df = df.rename(columns={"Temperature - (Celsius)" : "temp"})
    df = df[["temp"]]
    time = list(range(len(df)))
    df["time"] = time

    return df

def get_years():
    start = int(input("Input start year of analysis \n"))
    end = int(input("Input end year of analysis \n"))

    # returns a tuple
    return start, end

def analysis(years, df):
    df_filtered = df.loc[f"{years[0]}":f"{years[1]}"]

    model = LinearRegression()
    model.fit(df_filtered[["time"]], df_filtered["temp"])
    preds = model.predict(df_filtered[["time"]])
    df_filtered["trend"] = preds

    ax = df_filtered.drop(columns="time").plot(figsize=(20, 4))
    ax.set_title(f"Linear trend (coef = {float(model.coef_[0] * 12)} °C per year)")
    plt.show()


if __name__ == "__main__":
    df = read_data()
    years = get_years()
    analysis(years, df)
