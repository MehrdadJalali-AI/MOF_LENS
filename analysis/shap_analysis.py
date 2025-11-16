
# analysis/shap_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def shap_analysis(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("results/plots/shap_summary.png", bbox_inches='tight', dpi=300)
    plt.close()

    pd.DataFrame(shap_values, columns=X_test.columns).to_csv("results/shap_analysis.csv")
    print("SHAP analysis → results/plots/shap_summary.png")
