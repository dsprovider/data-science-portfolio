# Imported Libraries
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# =====================================================================================================================

def main():

    print(">> Starting the execution...")
    file_path = os.path.join(os.getcwd(), "houses_Madrid.csv")

    df = pd.read_csv(file_path, encoding="utf-8")
    print(">> Data loaded successfully!")

    # -- 1. FEATURES PREPARATION ---------------------------
    # a) Numerical Features:
    # [+] sq_mt_built
    # [+] n_rooms
    # [+] n_bathrooms

   # Fill NaN values with the median for 'sq_mt_built'
    sq_mt_built_median_value = df['sq_mt_built'].median()
    df['sq_mt_built_encoded'] = df['sq_mt_built'].fillna(sq_mt_built_median_value)

    # Fill NaN values with the median for 'n_bathrooms'
    n_bathrooms_median_value = df['n_bathrooms'].median()
    df['n_bathrooms_encoded'] = df['n_bathrooms'].fillna(n_bathrooms_median_value)

    
    # b) Binary Features
    # [+] is_new_development
    # [+] is_renewal_needed
    # [+] has_central_heating
    # [+] has_individual_heating
    # [+] has_lift
    # [+] has_private_parking
    # [+] has_garden

    # c) Categorical Features (discarded for complexity reasons)
    # [+] energy_certificate
    # [+] neighborhood_id
    

    # -- 2. BINARY ENCODING ----------------------------------
    df['is_new_development_encoded'] = df['is_new_development'].fillna(0).astype(int)
    df['is_renewal_needed_encoded'] = df['is_renewal_needed'].astype(int) # True --> 1; False --> 0
    df['has_central_heating_encoded'] = df['has_central_heating'].fillna(2).astype(int) # True --> 1; False --> 0; NaN --> 2
    df['has_individual_heating_encoded'] = df['has_individual_heating'].fillna(2).astype(int) # True --> 1; False --> 0; NaN --> 2
    df['has_lift_encoded'] = df['has_lift'].fillna(0).astype(int)
    df['has_garden_encoded'] = df['has_garden'].fillna(0).astype(int)

    selected_features = [
        'sq_mt_built_encoded', 'n_rooms', 'n_bathrooms_encoded', 'is_new_development_encoded', 'is_renewal_needed_encoded',
        'has_central_heating_encoded', 'has_individual_heating_encoded', 'has_lift_encoded', 'has_garden_encoded']    
    simplified_df = df[selected_features]
    print(">> Feature preparation completed!")


    # -- 3. SPLIT DATASET ----------------------------------
    X = simplified_df # features
    y = df['buy_price'] # target (price)

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(">> Dataset split completed!")


    # -- 4. TRAIN KNN MODEL -----------------------------
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    print(">> KNN model training completed!")


    # -- 5. EVALUATE KNN MODEL --------------------------
    knn_mse = mean_squared_error(y_test, knn_predictions)
    knn_r2 = r2_score(y_test, knn_predictions)
    print(f"\t[+] KNN Model: MSE = {knn_mse}, R² = {knn_r2}")


    # -- 6. TRAIN DECISION TREE MODEL -------------------
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    tree_predictions = decision_tree_model.predict(X_test)
    print(">> Decision Tree model training completed!")


    # -- 7. EVALUATE DECISION TREE MODEL ----------------
    tree_mse = mean_squared_error(y_test, tree_predictions)
    tree_r2 = r2_score(y_test, tree_predictions)
    print(f"\t[+] Decision Tree Model: MSE = {tree_mse}, R² = {tree_r2}")

    # -- 8. RESULTS --------------------------------------
    # KNN Model: MSE = 145474787560.4482, R² = 0.7427745191166459
    # Decision Tree Model: MSE = 180265853473.17615, R² = 0.6812576830385915

    # -- 9. PREDICTION FOR NEW HOUSE ----------------------------------
    print(">> Making predictions for a new house...")

     # New house features (example values)
    new_house_features = {
        'sq_mt_built_encoded': 100,  # Example: 100 square meters
        'n_rooms': 3,        # Example: 3 rooms
        'n_bathrooms_encoded': 2,    # Example: 2 bathrooms
        'is_new_development_encoded': 1,  # Example: is new development
        'is_renewal_needed_encoded': 0,     # Example: does not need renewal
        'has_central_heating_encoded': 1,   # Example: has central heating
        'has_individual_heating_encoded': 0, # Example: does not have individual heating
        'has_lift_encoded': 1,               # Example: has lift
        'has_garden_encoded': 0               # Example: does not have garden
    }

     # Convert the new house features into DataFrame
    new_house_df = pd.DataFrame([new_house_features])
    
    # Make predictions
    knn_new_prediction = knn_model.predict(new_house_df)
    tree_new_prediction = decision_tree_model.predict(new_house_df)

    print(f">> Predicted price for the new house (KNN): {knn_new_prediction[0]:.2f}")
    print(f">> Predicted price for the new house (Decision Tree): {tree_new_prediction[0]:.2f}")

    print(">> Execution completed!")

if __name__ == "__main__":
    main()