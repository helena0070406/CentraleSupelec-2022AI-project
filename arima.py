import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima 
from sklearn.preprocessing import MinMaxScaler 

def MAE(pred, true):
	return np.mean(np.absolute(pred-true))

def MAPE(pred, true):
	return np.mean(np.absolute(np.divide((true - pred), true)))

def RMSE(pred, true):
	return np.sqrt(np.mean(np.square(pred-true)))

data = np.load(r'PeMS04.npz')


flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis] 
true = []
predict = [] 
node_num = len(flow_data[:,0,0]) 
for node in range(node_num):
	node_data = flow_data[node, 13536:16992, :] 
	node_data = pd.DataFrame(node_data) 
	node_data.columns = list('Y')

	scaler = MinMaxScaler()
	node_data['Y'] = scaler.fit_transform(node_data)


	history_len = 24
	predict_len = 1 
	history = node_data['Y']
 
for t in range(node_data['Y'].shape[0]-history_len-predict_len+1):
    inputs = history[t:history_len+t]
    model = auto_arima(inputs, start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           seasonal=True,
                           d=None, D=1, trace=False,
                           error_action='ignore',
                           suppress_warnings=True) 
model.fit(inputs) 
y_hat = model.predict(1) 


y_hat = scaler.inverse_transform(y_hat[:, np.newaxis]) 
predict.append(y_hat) 

a = np.array(history[history_len+t:history_len+t+predict_len])
a = scaler.inverse_transform(a[:, np.newaxis])
true.append(a)

prediction = np.array(predict)
truth = np.array(true)
prediction = prediction[:, np.newaxis]
truth = truth[:, np.newaxis]
ARIMA_mape = MAPE(prediction, truth)
ARIMA_mae = MAE(prediction, truth)
ARIMA_rmse = RMSE(prediction, truth)

print(ARIMA_mape, ARIMA_mae, ARIMA_rmse)



  

