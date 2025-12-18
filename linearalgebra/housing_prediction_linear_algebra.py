"""
California Housing Price Prediction Using Linear Algebra
Simplified Version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("=" * 70)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("Linear Algebra Implementation")
    print("=" * 70)
    
    # 1. LOAD DATA
    print("\n1. Loading California Housing dataset...")
    data = fetch_california_housing()
    X_raw = data.data
    y_raw = data.target.reshape(-1, 1)
    feature_names = data.feature_names
    
    print(f"   • Samples: {X_raw.shape[0]}")
    print(f"   • Features: {X_raw.shape[1]}")
    
    # Select 3 features for simplicity
    selected_features = ['MedInc', 'AveRooms', 'HouseAge']
    selected_indices = [feature_names.index(f) for f in selected_features]
    X_selected = X_raw[:, selected_indices]
    print(f"   • Selected features: {selected_features}")
    
    # 2. SPLIT AND STANDARDIZE
    print("\n2. Splitting and standardizing data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_selected, y_raw, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)
    
    # Add bias column
    X_train = np.c_[np.ones((X_train_std.shape[0], 1)), X_train_std]
    X_test = np.c_[np.ones((X_test_std.shape[0], 1)), X_test_std]
    
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Test samples: {X_test.shape[0]}")
    
    # 3. NORMAL EQUATION
    print("\n3. Solving Normal Equation: w = (XᵀX)⁻¹Xᵀy")
    
    # Step by step calculation
    X_T = X_train.T
    X_T_X = X_T @ X_train
    X_T_X_inv = np.linalg.inv(X_T_X)
    X_T_y = X_T @ y_train
    w = X_T_X_inv @ X_T_y
    
    print(f"   • Xᵀ shape: {X_T.shape}")
    print(f"   • XᵀX shape: {X_T_X.shape}")
    print(f"   • (XᵀX)⁻¹ shape: {X_T_X_inv.shape}")
    
    # Display weights
    print("\n   Optimal weights found:")
    print(f"   Bias (w₀): {w[0,0]:.6f}")
    for i, feature in enumerate(selected_features):
        print(f"   {feature} (w{i+1}): {w[i+1,0]:.6f}")
    
    # 4. PREDICTIONS
    print("\n4. Making predictions...")
    y_train_pred = X_train @ w
    y_test_pred = X_test @ w
    
    # 5. EVALUATION
    print("\n5. Model Performance:")
    
    # Training metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n   Training Set:")
    print(f"   • MSE:  {train_mse:.4f}")
    print(f"   • RMSE: {train_rmse:.4f} (${train_rmse*100000:,.0f})")
    print(f"   • R²:   {train_r2:.4f} ({train_r2*100:.1f}% variance explained)")
    
    print(f"\n   Test Set:")
    print(f"   • MSE:  {test_mse:.4f}")
    print(f"   • RMSE: {test_rmse:.4f} (${test_rmse*100000:,.0f})")
    print(f"   • R²:   {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
    
    # 6. SAMPLE PREDICTIONS
    print("\n6. Sample predictions (first 5 test samples):")
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error ($)':<15}")
    print("-" * 65)
    
    for i in range(5):
        actual = y_test[i, 0]
        predicted = y_test_pred[i, 0]
        error = predicted - actual
        error_dollars = error * 100000
        
        print(f"{i:<8} ${actual:<11.3f} ${predicted:<11.3f} {error:<12.3f} ${error_dollars:>10,.0f}")
    
    # 7. VISUALIZATION
    print("\n7. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price ($100,000s)')
    axes[0, 0].set_ylabel('Predicted Price ($100,000s)')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($100,000s)')
    axes[0, 1].set_ylabel('Residuals ($100,000s)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature Weights
    feature_labels = ['Bias'] + selected_features
    weights = w.flatten()
    colors = ['red', 'blue', 'green', 'orange']
    axes[1, 0].bar(feature_labels, weights, color=colors)
    axes[1, 0].set_ylabel('Weight Value')
    axes[1, 0].set_title('Model Weights (Feature Importance)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (label, weight) in enumerate(zip(feature_labels, weights)):
        axes[1, 0].text(i, weight, f'{weight:.3f}', 
                       ha='center', va='bottom' if weight >= 0 else 'top')
    
    # Plot 4: Error Distribution
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error ($100,000s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('housing_results.png', dpi=150, bbox_inches='tight')
    print("   • Plot saved as 'housing_results.png'")
    
    # 8. COMPARE WITH SKLEARN
    print("\n8. Comparison with scikit-learn's LinearRegression:")
    from sklearn.linear_model import LinearRegression
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_std, y_train)
    
    print(f"\n   Our Model Weights:")
    print(f"   Bias: {w[0,0]:.6f}")
    for i, feature in enumerate(selected_features):
        print(f"   {feature}: {w[i+1,0]:.6f}")
    
    print(f"\n   Sklearn Model Weights:")
    print(f"   Bias: {sklearn_model.intercept_[0]:.6f}")
    for i, feature in enumerate(selected_features):
        print(f"   {feature}: {sklearn_model.coef_[0,i]:.6f}")
    
    # 9. FINAL MODEL FORMULA
    print("\n9. Final Model Formula:")
    print(f"\n   Price = {w[0,0]:.4f} + {w[1,0]:.4f}×MedInc_std + {w[2,0]:.4f}×AveRooms_std + {w[3,0]:.4f}×HouseAge_std")
    
    print(f"\n   Where standardized features are:")
    for i, feature in enumerate(selected_features):
        mean = scaler.mean_[i]
        std = scaler.scale_[i]
        print(f"   {feature}_std = ({feature} - {mean:.3f}) / {std:.3f}")
    
    # 10. SAVE RESULTS
    print("\n10. Saving results...")
    
    # Save weights
    weights_df = pd.DataFrame({
        'feature': feature_labels,
        'weight': weights,
        'interpretation': [
            'Base price when features are average',
            'Price change per std increase in income',
            'Price change per std increase in rooms',
            'Price change per std increase in house age'
        ]
    })
    weights_df.to_csv('model_weights.csv', index=False)
    print("   • Weights saved to 'model_weights.csv'")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test.flatten(),
        'predicted': y_test_pred.flatten(),
        'error': residuals.flatten(),
        'actual_dollars': y_test.flatten() * 100000,
        'predicted_dollars': y_test_pred.flatten() * 100000,
        'error_dollars': residuals.flatten() * 100000
    })
    predictions_df.to_csv('housing_predictions.csv', index=False)
    print("   • Predictions saved to 'housing_predictions.csv'")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nKey Insights:")
    print(f"1. Income is the strongest predictor (weight: {w[1,0]:.3f})")
    print(f"2. Model explains {test_r2*100:.1f}% of price variation")
    print(f"3. Average prediction error: ${test_rmse*100000:,.0f}")
    print(f"4. Base California house price: ${w[0,0]*100000:,.0f}")
    
    plt.show()

if __name__ == "__main__":
    main()