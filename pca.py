import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    df_heart = pd.read_csv('./data/heart.csv')

    #print(df_heart.head(5))

    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    df_features_scaled = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features_scaled, df_target, test_size=0.3, random_state=42)

    pca = PCA(n_components=3)

    pca.fit(X_train)

    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)

    logistic.fit(df_train,y_train)

    print('SCORE PCA: ', logistic.score(df_test, y_test))

