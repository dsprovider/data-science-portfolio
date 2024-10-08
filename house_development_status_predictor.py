# Imported Libraries
import os
import pprint
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================================================================

def main():

    print(">> Starting the execution...")
    file_path = os.path.join(os.getcwd(), "houses_Madrid.csv")

    df = pd.read_csv(file_path, encoding="utf-8")
    print(">> Data loaded successfully!")

    # -- 1. FEATURES PREPARATION ---------------------------
    # a) Numerical Features:
    # [+] sq_mt_built
    # [+] sq_mt_useful
    # [+] n_rooms
    # [+] n_bathrooms
    # [+] buy_price

   # Fill NaN values with the median for 'sq_mt_built'
    sq_mt_built_median_value = df['sq_mt_built'].median()
    df['sq_mt_built_encoded'] = df['sq_mt_built'].fillna(sq_mt_built_median_value)

    # Fill NaN values with the median for 'sq_mt_useful'
    sq_mt_useful_median_value = df['sq_mt_useful'].median()
    df['sq_mt_useful_encoded'] = df['sq_mt_useful'].fillna(sq_mt_useful_median_value)

    # Fill NaN values with the median for 'n_bathrooms'
    n_bathrooms_median_value = df['n_bathrooms'].median()
    df['n_bathrooms_encoded'] = df['n_bathrooms'].fillna(n_bathrooms_median_value)

    # b) Binary Features
    # [+] has_central_heating
    # [+] has_individual_heating
    # [+] has_lift
    # [+] has_garden

    # -- 2. BINARY ENCODING ----------------------------------
    df['is_new_development_encoded'] = df['is_new_development'].fillna(0).astype(int)
    df['has_central_heating_encoded'] = df['has_central_heating'].fillna(2).astype(int) # True --> 1; False --> 0; NaN --> 2
    df['has_individual_heating_encoded'] = df['has_individual_heating'].fillna(2).astype(int) # True --> 1; False --> 0; NaN --> 2
    df['has_lift_encoded'] = df['has_lift'].fillna(0).astype(int)
    df['has_garden_encoded'] = df['has_garden'].fillna(0).astype(int)

    selected_features = [
        'sq_mt_built_encoded', 'sq_mt_useful_encoded', 'n_rooms', 'n_bathrooms_encoded', 
        'has_central_heating_encoded', 'has_individual_heating_encoded',
        'has_lift_encoded', 'has_garden_encoded', 'buy_price'
    ]    
    simplified_df = df[selected_features]
    print(">> Feature preparation completed!")


    # -- 3. SPLIT DATASET ----------------------------------
    X = simplified_df # features
    y = df['is_new_development_encoded'] # target

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(">> Dataset split completed!")


    # -- 4. SCALE NUMERICAL FEATURES -------------------------
    scaler = StandardScaler()
    
    # Define the numerical features to be scaled
    numerical_features = ['sq_mt_built_encoded', 'sq_mt_useful_encoded', 'n_rooms', 'n_bathrooms_encoded', 'buy_price']
    
    # Fit scaler on training data and transform both training and testing data
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    print(">> Feature scaling completed!")

    # -- 5. TRAIN LOGISTIC MODEL -----------------------------
    logreg = LogisticRegression(random_state=16, max_iter=500)  # Instantiate the logistic regression model
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(">> Logistic model training completed!")


    # -- 6. EVALUATE LOGISTIC MODEL --------------------------
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    # -- 7. PREDICTION FOR NEW HOUSE ----------------------------------
    print(">> Making predictions for a new house...")

    # New house features (example values)
    new_house_features = {
        'sq_mt_built_encoded': 100,  # Example: 100 square meters
        'sq_mt_useful_encoded': 90,  # Example: 90 square meters useful space
        'n_rooms': 3,                # Example: 3 rooms
        'n_bathrooms_encoded': 2,     # Example: 2 bathrooms
        'has_central_heating_encoded': 1,   # Example: has central heating
        'has_individual_heating_encoded': 0, # Example: does not have individual heating
        'has_lift_encoded': 1,               # Example: has lift
        'has_garden_encoded': 0,             # Example: does not have garden
        'buy_price': 300000                 # Example: 300,000 buy price
    }

    # Make predictions  
    new_house_df = pd.DataFrame([new_house_features]) # Convert the new house features into DataFrame
    new_house_df = new_house_df[simplified_df.columns] # Ensure the order of columns matches the training data
    new_prediction = logreg.predict(new_house_df)

    if new_prediction[0] == 1:
        print(">> The new house is predicted to be a new development.")
    else:
        print(">> The new house is predicted to be an old development.")

    print(">> Execution completed!")

if __name__ == "__main__":
    main()