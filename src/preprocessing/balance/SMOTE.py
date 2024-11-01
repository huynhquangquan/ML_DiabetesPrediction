from imblearn.over_sampling import SMOTE
import pandas as pd

def balance_train(df, cat_feature):
    smote = SMOTE(random_state=42)
    X = df.drop(columns=[cat_feature])
    y = df[cat_feature]
    X_train, y_train = smote.fit_resample(X,y)

    # Merge X and y to a single Dataframe
    balanced_train = pd.concat([pd.DataFrame(X_train,columns=X.columns), pd.DataFrame(y_train, columns=['Outcome'])],axis=1)

    return balanced_train