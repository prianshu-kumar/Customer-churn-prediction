import pandas as pd


def create_features(df):

    df = df.copy()

    # Average spend per tenure
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] / (df["tenure"] + 1)
    )

    # Long term customer
    df["LongTermCustomer"] = (
        df["tenure"] > 24
    ).astype(int)

    # High monthly charges
    df["HighMonthlyCharges"] = (
        df["MonthlyCharges"] > 80
    ).astype(int)

    return df