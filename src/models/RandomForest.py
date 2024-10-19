from src import utilities
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # Split train data
    base_dir = Path(__file__).parent.parent

    check_processed = bool(utilities.check_processed())

    if check_processed is False:
        sys.exit()

    train = pd.read_csv(base_dir / '..' / 'data' / 'processed' / 'train.csv')

    X_train = train.drop(columns=['Outcome'])
    y_train = train['Outcome']

    class_weights={0:1,1:2}
    model = RandomForestClassifier(n_estimators=120, random_state=42,class_weight=class_weights)
    model.fit(X_train, y_train)

    utilities.joblib_dump(model, 'random_forest')

    print("-------Lưu mô hình Random Forest hoàn tất")