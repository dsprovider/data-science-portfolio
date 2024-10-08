# 📊 Data Science Portfolio

Welcome to my **Data Science Portfolio**! 🌟 This repository is where I share the data science and machine learning projects I am working on as I continue to learn and grow in this exciting field. While I am not an expert (yet!), these projects reflect my journey in applying data-driven techniques to solve real-world problems.

Each project covers different aspects of data science, from data exploration and preprocessing to building and evaluating predictive models. So far, you will find:

# 🏠 **Housing Price Prediction**
This project involves creating a regression model to predict house prices based on various features such as square meters, heating options, and garden availability.
  1. **Script:** house_price_predictor.py
  2. **Dataset:** https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market
  3. **Models used:** **K-Nearest Neighbors (KNN)** and **Decision Tree Regression**.
  4. **Input Features:**
     * sq_mt_built
     * n_rooms
     * n_bathrooms
     * is_new_development
     * is_renewal_needed
     * has_central_heating
     * has_individual_heating
     * has_lift
     * has_private_parking
     * has_garden
  5. **Target Feature:** buy_price 
  6. **Note:** for simplicity, the model currently does not consider categorical parameters such as neighborhood or energy certificate. Future improvements should aim to incorporate these factors into the code.

# 🏢 **Apartment Condition Classification**
This project uses logistic regression to predict whether a house is a new or old development based on various features such as size, number of rooms, bathrooms, and additional amenities.
1. **Script:** house_development_status_predictor.py
2. **Dataset:** https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market
3. **Models used:** Logistic Regression
4. **Input Features:**
   * sq_mt_built
   * sq_mt_useful
   * n_rooms
   * n_bathrooms
   * has_central_heating
   * has_individual_heating
   * has_lift
   * has_garden
   * buy_price
6. **Target Feature:** is_new_development
7. **Note:** the dataset exhibits a significant class imbalance, with a majority of instances representing old developments. As a result, the model tends to be biased towards predicting the majority class (old houses), leading to a failure in accurately detecting the minority class (new developments). Future improvements to the code should incorporate techniques to address this.

Feel free to explore the code and maybe even pick up a few ideas for your own projects! 🚀
