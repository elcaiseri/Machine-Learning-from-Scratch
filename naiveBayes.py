import numpy as np

class NaiveBayes:
    def __init__(self) -> None:
        self.classes = None
    
    def fit(self, X, y):
        n_samples, featuers = X.shape # (samples, features)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, featuers))
        self.var = np.zeros((n_classes, featuers))
        
        self.Py = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            y_mask = (y == c)
            X_ = X[y_mask, :]
            
            self.mean[c, :] = X_.mean(axis=0) # (classes, featuers)
            self.var[c, :] = X_.var(axis=0) # (classes, featuers)
            self.Py[c] = X_.shape[0] / n_samples
                
    def _pdf(self, x, mean, var):
        c1 = np.sqrt(2 * np.pi * var)
        c2 = np.exp( -(x - mean) ** 2 / (2 * var))
        return np.log(c2 / c1)
    
    def predict(self, X):     
        Pyx = []
        for c in self.classes:
            pyx = np.sum(self._pdf(X, self.mean[c, :], self.var[c, :]), axis=1) + np.log(self.Py[c])
            Pyx.append(pyx)
        return np.argmax(Pyx, 0)
    

if __name__ == "__main__":
    # imports
    from sklearn.model_selection import train_test_split as tts 
    from sklearn import datasets as ds 
    import matplotlib.pyplot as plt
    
    # create dataset
    np.random.seed(123)    
    X, y = ds.make_classification(n_samples=1000, n_features=10, n_classes=2)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    
    # model
    lr = NaiveBayes()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # evaluate
    acc = np.mean((y_test == y_pred))  
    print("Naive Bayes Test Accuracy:", acc)