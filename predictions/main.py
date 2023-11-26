import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


tf.random.set_seed(
    1
)
tf.keras.utils.set_random_seed(
    1
)

df_train_traffick = pd.read_csv('train_traffick.csv')
df_test_traffick = pd.read_csv('test_traffick.csv')

region_mapping = {
    'Anenii Noi': 0.15,
    'Basarabeasca': 0.08,
    'Balti': 0.25,
    'Briceni': 0.05,
    'Cahul': 0.12,
    'Cantemir': 0.18,
    'Calarasi': 0.20,
    'Causeni': 0.07,
    'Chisinau': 0.30,
    'Cimislia': 0.14,
    'Comrat': 0.10,
    'Criuleni': 0.06,
    'Donduseni': 0.22,
    'Drochia': 0.11,
    'Dubasari': 0.16,
    'Edinet': 0.09,
    'Falesti': 0.21,
    'Floresti': 0.13,
    'Glodeni': 0.17,
    'Hincesti': 0.08,
    'Ialoveni': 0.26,
    'Leova': 0.14,
    'Nisporeni': 0.10,
    'Ocnita': 0.05,
    'Orhei': 0.23,
    'Rezina': 0.16,
    'Riscani': 0.12,
    'Singerei': 0.27,
    'Soroca': 0.06,
    'Straseni': 0.19,
    'Soldanesti': 0.11,
    'Stefan Voda': 0.24,
    'Taraclia': 0.15,
    'Telenesti': 0.08,
    'Tighina': 0.14,
    'Tiraspol': 0.10,
    'Ungheni': 0.25
}

df_train_traffick.loc[df_train_traffick['region_name'].isin(region_mapping.keys()), 'chance_of_human_trafficking'] = df_train_traffick['region_name'].map(region_mapping)

print(df_train_traffick)

selected_features_traffick = ['demographics','economic_conditions','previous_trafficking_incidents','social_factors','airport','train_station','seaport','bus_station', 'chance_of_human_trafficking']

df_selected_train_traffick = df_train_traffick[selected_features_traffick]

df_selected_train_traffick['economic_conditions'] = df_selected_train_traffick['economic_conditions'].map({'Very Poor': 1, 'Poor': 0.75, 'Moderate': 0.5,'Good': 0.25, 'Excellent': 0})

df_selected_train_traffick['social_factors'] = df_selected_train_traffick['social_factors'].map({'Low': 1, 'Moderate': 0.5, 'High': 0})

X_train = df_selected_train_traffick.drop('chance_of_human_trafficking', axis=1)
y_train = df_selected_train_traffick['chance_of_human_trafficking']

model = LinearRegression()
model.fit(X_train, y_train)
y_predictions_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_predictions_train)
print(f'Ordinary Least Squares: Mean Squared Error on Train Set: {mse_train}')

df_test_traffick['economic_conditions'] = df_test_traffick['economic_conditions'].map({'Very Poor': 1, 'Poor': 0.75, 'Moderate': 0.5,'Good': 0.25, 'Excellent': 0})

df_test_traffick['social_factors'] = df_test_traffick['social_factors'].map({'Low': 1, 'Moderate': 0.5, 'High': 0})

X_test = df_test_traffick[selected_features_traffick[:-1]]  # Exclude 'Chance of Admit' for testing

y_predictions_test = model.predict(X_test)

print("\nLinear Regression Predictions:")
print(y_predictions_test)

model_nn = Sequential()
model_nn.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model_nn.add(Dense(units=1))
model_nn.summary()

opt = keras.optimizers.Adam(learning_rate=0.00001)
model_nn.compile(loss='mean_squared_error', optimizer=opt)
history = model_nn.fit(X_train, y_train, epochs=1500, batch_size=32, validation_split=0.2)

y_predictions_train_nn = model_nn.predict(X_train)
mse_train_nn = mean_squared_error(y_train, y_predictions_train_nn)
print(f'Mean Squared Error on Train Set (Neural Network): {mse_train_nn}')

y_predictions_test_nn = model_nn.predict(X_test)

print("\nLinear Regression Predictions:")
print(y_predictions_test)

print("\nNeural Network Predictions:")
print(y_predictions_test_nn)

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_predictions_test)), y_predictions_test, label='OLS Predictions (Test Set)')
plt.scatter(range(len(y_predictions_test_nn)), y_predictions_test_nn, label='Neural Network Predictions (Test Set)')

plt.title('Predictions on Test Set')
plt.xlabel('Data Points')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# y_predictions_mapped = []

# for region in y_predictions_test_nn:
#     try:
#         mapped_value = region_mapping[region[0]]
#         y_predictions_mapped.append([mapped_value])
#     except KeyError:
#         print(f"KeyError: Region '{region[0]}' not found in region_mapping. Please update region_mapping.")

# # Print the result
# print(y_predictions_mapped)

print("\n\nNeural Network Predictions:")

region_mapping = {
    'Anenii Noi': y_predictions_test_nn[0][0],
    'Basarabeasca': y_predictions_test_nn[1][0],
    'Balti': y_predictions_test_nn[2][0],
    'Briceni': y_predictions_test_nn[3][0],
    'Cahul': y_predictions_test_nn[4][0],
    'Cantemir': y_predictions_test_nn[5][0],
    'Calarasi': y_predictions_test_nn[6][0],
    'Causeni': y_predictions_test_nn[7][0],
    'Chisinau': y_predictions_test_nn[8][0],
    'Cimislia': y_predictions_test_nn[9][0],
    'Comrat': y_predictions_test_nn[10][0],
    'Criuleni': y_predictions_test_nn[11][0],
    'Donduseni': y_predictions_test_nn[12][0],
    'Drochia': y_predictions_test_nn[13][0],
    'Dubasari': y_predictions_test_nn[14][0],
    'Edinet': y_predictions_test_nn[15][0],
    'Falesti': y_predictions_test_nn[16][0],
    'Floresti': y_predictions_test_nn[17][0],
    'Glodeni': y_predictions_test_nn[18][0],
    'Hincesti': y_predictions_test_nn[19][0],
    'Ialoveni': y_predictions_test_nn[20][0],
    'Leova': y_predictions_test_nn[21][0],
    'Nisporeni': y_predictions_test_nn[22][0],
    'Ocnita': y_predictions_test_nn[23][0],
    'Orhei': y_predictions_test_nn[24][0],
    'Rezina': y_predictions_test_nn[25][0],
    'Riscani': y_predictions_test_nn[26][0],
    'Singerei': y_predictions_test_nn[27][0],
    'Soroca': y_predictions_test_nn[28][0],
    'Straseni': y_predictions_test_nn[29][0],
    'Soldanesti': y_predictions_test_nn[30][0],
    'Stefan Voda': y_predictions_test_nn[31][0],
    'Taraclia': y_predictions_test_nn[32][0],
    'Telenesti': y_predictions_test_nn[33][0],
    'Tighina': y_predictions_test_nn[34][0],
    'Tiraspol': y_predictions_test_nn[35][0],
    'Ungheni': y_predictions_test_nn[36][0]
}

# print(region_mapping)

for region, value in region_mapping.items():
    print(f'{region:<20} {value:.15f}')
