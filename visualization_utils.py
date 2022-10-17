import matplotlib.pyplot as plt
def plot_train_test_pred(train_y, test_y, pred):
    """
    Args:
    - train_y = array of Close price in training set
    - test_y = array of Close price in test set (true values)
    - pred = array of predicted Close price (from test set)
    """
    plt.figure(figsize = (30, 10))
    plt.plot(train_y, label = 'Training')
    plt.plot(test_y, label = 'Testing')
    plt.plot(pred, label = 'Prediction')
    plt.legend(loc = 'upper right')
    plt.show()