import numpy as np

print("Hello World!")

class KNN:
    def __init__(self, k) -> None:
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        dis = np.argsort([self._predict_point(x) for x in X], axis=1)
        yy = self.y[dis.T][:self.k].T
        return [np.argmax(np.bincount(yyy)) for yyy in yy]
        
    def _predict_point(self, p):
        return np.sum(np.sqrt((self.X - p) ** 2), axis=1)

if __name__ == "__main__":
    # imports
    from sklearn.model_selection import train_test_split as tts 
    from sklearn import datasets as ds 
    import matplotlib.pyplot as plt
    
    
    # create dataset
    np.random.seed(1234)   
    bc = ds.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=1234)
    
    # model
    lr = KNN(10)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # evaluate
    acc = (y_test == y_pred).mean()
    print("KNN Test Accuracy:", acc)