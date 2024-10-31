from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def balance_train(df,classification_feature):
    rus = RandomUnderSampler(random_state=42)
    X = df.drop(columns=[classification_feature])
    y = df[classification_feature]

    rus = RandomUnderSampler(random_state=42)

    X_res, y_res = rus.fit_resample(X,y)

    df_res = pd.concat([X_res,y_res],axis=1)
    return df_res