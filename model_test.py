import joblib
#载入gbr.pkl模型
model = joblib.load('models_gbr/gbr_model_0.pkl')
print(model.predict([[3.8,1.4,0.4,0.1,0.2,0.1,0.1]]))
