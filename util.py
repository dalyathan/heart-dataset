import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def save_cm(model_type,model_name,cm, show=False):
    plt.figure(figsize=(9.2, 6))
    class_names = ['No Death Event','Death Event']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {model_name}", )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.savefig(f"./confusion_matrices/{model_type}/{model_name}.png")
    if not show:
        plt.close()

def read_data():
    df = pd.read_csv('./heart_failure_clinical_records_dataset-1-1.csv')
    df = df.drop_duplicates()
    df = df.fillna(df.median())
    continous_features = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine', 'time']  
    outliers(df[continous_features],df, True)
    feature_names=df.columns[:-1]
    features = df.drop('DEATH_EVENT', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['DEATH_EVENT'] = df['DEATH_EVENT'].values
    X = df_scaled.drop('DEATH_EVENT', axis=1)
    y = df_scaled['DEATH_EVENT']
    return X, y, feature_names

def outliers(df_out,data, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.)
        Q3 = np.percentile(feature_data, 75.)
        IQR = Q3-Q1
        outlier_step = IQR * 1
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            data.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))