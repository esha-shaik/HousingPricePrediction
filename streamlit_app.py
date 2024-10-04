import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Housing Price Prediction Project", layout="wide")

# Title
st.title("Housing Price Prediction Project")

# Introduction & Background
st.header("Introduction and Background")
st.subheader("_Literature Review_")
st.write("""
Housing prices are unpredictable especially due to the many factors impacting the prices of houses. Because of unpredictable housing prices, a machine learning model would be beneficial in determining how housing prices may change. 

Adetunji et al. (2022) discuss their RandomForest model that predicts housing prices. They identified significant factors that play a role in housing prices such as crime rate, number of rooms, and accessibility to highways to approximate how high the housing price would be based on these attributes (Adetunji et al., 2022, p. 808-812). 

Vineeth et al. (2018) looks at different algorithms like simple linear regression, multiple linear regression, and neural networks for predicting housing prices. The researchers examined factors such as price, number of bedrooms/bathrooms, square feet, and etc. From these algorithms, it was found that the neural networks perform the best (Vineeth et al., 2018).

Gupta et al. (2022) applied a Bayesian dynamic factor model to analyze housing market trends by assessing how macroeconomic uncertainty influences state housing prices. The model found that national factors accounted for 35% of housing price growth. Additionally, a random forest model helped identify key drivers of macroeconomic uncertainty and predict the best-fit model for national house prices.
""")

# Dataset Description
st.header("Dataset Description")
st.subheader("_NY Housing_")
st.write("The dataset includes prices for 4,800 New York houses with features like house type, size, and location. It helps analyze how these factors influence pricing but lacks historical data for predicting trends over time.")
st.subheader("_California Housing_")
st.write("This dataset covers 20,640 California houses with features like house age, median income, and location. It enables analysis of how income and house characteristics affect pricing.")
st.subheader("_Dataset Links_")
st.markdown("- [NY Housing Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)")
st.markdown("- [California Housing Dataset](https://huggingface.co/datasets/leostelon/california-housing)")

# Problem
st.header("Problem Definition")
st.write("""
We aim to tackle the unpredictability of housing prices, which affects buyers, sellers, and developers. Housing data is often noisy and non-linear, so using machine learning models allows for more accurate price predictions. This approach enhances pricing transparency, helps stakeholders make better decisions, and reduces risks in the housing market caused by inaccurate forecasts.
""")


# Methods
st.header("Methods")
st.subheader("_Data Preprocessing_")
st.markdown("""
- scikit-learn StandardScaler to standardize numerical data and improve convergence of the machine learning models. TFIDF Vectorization for categorical and text data. 
- pandas Series.string to extract keywords from the type of house and address in the NYC data set.
- pandas DataFrames to merge the two datasets into one DataFrame. 
- geopy package Nominatim to convert addresses in the NYC data set into longitude and latitude points to match the California dataset.
- After merging, use sklearn haversine_distances to calculate distances to major city points, schools, and transportation to generate more insights.
""")

st.subheader("_Machine Learning Models_")
st.markdown("""
- Supervised learning LinearRegression model is applicable since we have clearly defined attributes (median household income, price, location, etc) and a target value (price of the house).
- Unsupervised models like K-means would uncover relations between attributes and help identify unique insights. We could use a hybrid stacking model by mixing this model with a supervised model.
- Supervised learning model RandomForest constructs several decision trees and returns an average prediction and ranks features based on importance in predictions; this is a robust solution for our nonlinear dataset.
""")

# Results and Discussion
st.header("Results and Discussion")
st.subheader("_Quantitative Metrics_")
st.write("""
- **Mean Absolute Error (MAE)**: Chosen because Linear Regression is a regression model, and MAE measures the average prediction error in the same unit as the target (dollars).
- **Root Mean Squared Error (RMSE)**: Emphasizes larger errors more than MAE.
- **R-squared (R²)**: Indicates the proportion of variance in house prices the Linear Regression model explains.
""")

st.subheader("_Project Goals_")
st.write("""
- **MAE**: Less than $10,000
- **RMSE**: Less than $15,000
- **R²**: Greater than 0.80
""")

st.subheader("_Expected Results_")
st.write("""
Linear Regression provides reliable price predictions, minimizing error and explaining most price variability, though it may struggle with non-linear relationships. K-means can capture more complex patterns, while RandomForest highlights the most influential features for predicting housing prices.
""")

# References
st.header("References")
st.write("""
Adetunji, A. B., Akande, O. N., Ajala, F. A., Oyewo, O., Akande, Y. F., & Oluwadara, G. (2022). House price prediction using random forest machine learning technique. _Procedia Computer Science, 199,_ 806-813.

Gupta, R., Marfatia, H. A., Pierdzioch, C., & Salisu, A. A. (2022). Machine learning predictions of housing market synchronization across US states: the role of uncertainty. _The Journal of Real Estate Finance and Economics,_ 1-23.
               
Vineeth, N., Ayyappa, M., & Bharathi, B. (2018). House price prediction using machine learning algorithms. _In Soft Computing Systems: Second International Conference, ICSCS 2018, Kollam, India, April 19–20, 2018, Revised Selected Papers 2_ (pp. 425-433). Springer Singapore.
""")

# Gantt Chart
st.header("Gantt Chart")
st.markdown("""
<iframe src="https://docs.google.com/spreadsheets/d/1cByuuEirIo4QXQRyprUYK4KuA0f8NmC5/edit?usp=sharing&ouid=118163818766847452393&rtpof=true&sd=true" width="100%" height="600"></iframe>
""", unsafe_allow_html=True)

# Contribution Table
st.header("Contribution Table")
contribution_data = {
    'Team Member': ['Ashmitha Aravind', 'Khushi Gupta', 'Natasha Setidadi', 'Vonesha Shaik', 'Chrystabel Sunata'],
    'Contribution': ['Worked on the problem statement and collaborated on the gantt chart.', 'Worked on metrics section, results and discussion. Edited the video.', 'Worked on the introduction and literature review.', 'Worked on methods preprocessing and models. Worked on the streamlit project.', 'Worked on methods preprocessing section with Esha. Worked on dataset descriptions.']
}
contribution_df = pd.DataFrame(contribution_data)
st.write(contribution_df)

# Video
st.header("Video")
st.write("Project overview:")
st.video("https://www.youtube.com/watch?v=3eKj4Z7qT6w&ab_channel=KhushiGupta")
