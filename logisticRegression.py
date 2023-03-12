import numpy as np

print("Hello world")

class LogisticRegression:
    def __init__(self, n_iter, learning_rate, th=0.5) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.th = th
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0
        
        for _ in range(self.n_iter):
            lr = (self.w @ X.T + self.b) / n
            yhat = self._sigmoid(lr)
                      
            db = 1 / n * np.sum(-y + yhat) 
            dw = 1 / n * (-y + yhat) @ X 
            
            self.b -= self.learning_rate * db
            self.w -= self.learning_rate * dw
                        
                    
    def predict(self, X):
        lr = self.b + self.w @ X.T
        return self._sigmoid(lr) > self.th 
    

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
    lr = LogisticRegression(1000, 0.01, 0.5)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # evaluate
    acc = (y_test == y_pred).mean()
    print("Logistic Regression Test Accuracy:", acc)