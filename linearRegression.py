import numpy as np

print("Hello world")

class LinearRegression:
    def __init__(self, n_iter, learning_rate) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0
        
        for _ in range(self.n_iter):
            yhat = (self.w @ X.T + self.b) / n
                                  
            db = 1 / n * np.sum(-y + yhat) 
            dw = 1 / n * (-y + yhat) @ X 
            
            self.b -= self.learning_rate * db
            self.w -= self.learning_rate * dw            
        
    def state_fit(self, X, y):
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = np.mean(y) - self.w * np.mean(X)
                    
    def predict(self, X):
        return self.b + self.w @ X.T
    

if __name__ == "__main__":
    # imports
    from sklearn.model_selection import train_test_split as tts 
    from sklearn import datasets as ds 
    
    # create dataset
    np.random.seed(42)    
    X, y = ds.make_regression(n_samples=10000, n_features=2, noise=5)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    
    # model
    lr = LinearRegression(100, 0.01)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # evaluate
    mse = np.mean((y_test - y_pred) ** 2)  
    print("Linear Regression Test MSE:", mse)