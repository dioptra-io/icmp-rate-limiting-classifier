from sklearn.preprocessing import MinMaxScaler

def minmax_scale(features):
    scaler = MinMaxScaler()
    scaler.fit(features)
    return scaler.transform(features)
