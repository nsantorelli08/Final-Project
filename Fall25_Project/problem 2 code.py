#Question 2 Code
print("\n" + "=" * 60)
print("QUESTION 2: Can weather predict next day's traffic?")
print("=" * 60)

dataset_2['NextDay_HighTemp'] = dataset_2['High Temp'].shift(-1)
dataset_2['NextDay_LowTemp'] = dataset_2['Low Temp'].shift(-1)
dataset_2['NextDay_Precipitation'] = dataset_2['Precipitation'].shift(-1)
dataset_2['NextDay_Total'] = dataset_2['Total'].shift(-1)


weather_data = dataset_2.dropna(subset=['NextDay_HighTemp', 'NextDay_LowTemp', 'NextDay_Precipitation', 'NextDay_Total'])

X_weather = weather_data[['NextDay_HighTemp', 'NextDay_LowTemp', 'NextDay_Precipitation']]
y_weather = weather_data['NextDay_Total']

X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

weather_model = LinearRegression()
weather_model.fit(X_train_scaled, y_train)

train_pred = weather_model.predict(X_train_scaled)
test_pred = weather_model.predict(X_test_scaled)

print("Weather Prediction Model Results:")
print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test R²: {r2_score(y_test, test_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred):.2f}")

precip_corr = weather_data['NextDay_Precipitation'].corr(weather_data['NextDay_Total'])
print(f"\nCorrelation between precipitation and traffic: {precip_corr:.4f}")
