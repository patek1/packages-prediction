# import the necessary packages including streamlit and PIL to create a GUI
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# import the function to transform the data and predict packages
from ml_for_gui import transform_data, predict_packages, relu

# import logo from the Swiss Post
image = Image.open("post_logo.png")
st.image(image, use_column_width=True)

# set title
st.title("Expected number of packages")
# display instruction for user
st.write("""#### Please insert your values""")

# create a list of months for the second slider "Weekday"
months = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
)

# create a list of weekdays for the second slider "Weekday"
weekdays = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
)

# create all interactive input possibilities for user such as the sliders
month = st.selectbox("Month", months)
weekday = st.selectbox("Weekday", weekdays)
total_pop = st.slider("Population size", 500, 20000, 10000, step=100)
mean_age = st.slider("Mean age", 35.0, 50.0, 40.0, step=0.1)
dist_to_city = st.slider("Distance to nearest city", 0, 50, 20)

# implement a solution to translate the weekday and month input into the necessary format for a prediction input - ugly ugh...
if month == "January":
    month_lst = [0,0,0,0,1,0,0,0,0,0,0,0]
if month == "February":
    month_lst = [0,0,0,1,0,0,0,0,0,0,0,0]
if month == "March":
    month_lst = [0,0,0,0,0,0,0,1,0,0,0,0]
if month == "April":
    month_lst = [1,0,0,0,0,0,0,0,0,0,0,0]
if month == "May":
    month_lst = [0,0,0,0,0,0,0,0,1,0,0,0]
if month == "June":
    month_lst = [0,0,0,0,0,0,1,0,0,0,0,0]
if month == "July":
    month_lst = [0,0,0,0,0,1,0,0,0,0,0,0]
if month == "August":
    month_lst = [0,1,0,0,0,0,0,0,0,0,0,0]
if month == "September":
    month_lst = [0,0,0,0,0,0,0,0,0,0,0,1]
if month == "Oktober":
    month_lst = [0,0,0,0,0,0,0,0,0,0,1,0]
if month == "November":
    month_lst = [0,0,0,0,0,0,0,0,0,1,0,0]
if month == "December":
    month_lst = [0,0,1,0,0,0,0,0,0,0,0,0]
if weekday == "Tuesday":
    day_lst = [1,0,0,0,0,0,0]
if weekday == "Thursday":
    day_lst = [0,1,0,0,0,0,0]
if weekday == "Friday":
    day_lst = [0,0,1,0,0,0,0]
if weekday == "Wednesday":
    day_lst = [0,0,0,1,0,0,0]
if weekday == "Monday":
    day_lst = [0,0,0,0,1,0,0]
if weekday == "Saturday":
    day_lst = [0,0,0,0,0,1,0]
if weekday == "Sunday":
    day_lst = [0,0,0,0,0,0,1]

# set color of calculation button to black
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #000000;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #000000;
    color:#ffffff;
    }
</style>""", unsafe_allow_html=True)

# create calculation button
go = st.button("Calculate number of packages")

# calculating the ML process and create prediction
packages = pd.read_csv("data/packages_final.csv", usecols=["mean_age", "total_pop", "month", "weekday", "packages", "distance_to_nearest_city"])

# using the transforming function to standardize the dataframe
X, y, X_data = transform_data(packages)

# standardize the input parameters
standardise_0 = lambda x: (x - X_data["mean_age"].mean())/X_data["mean_age"].std()
standardise_1 = lambda x: (x - X_data["total_pop"].mean())/X_data["total_pop"].std()
standardise_2 = lambda x: (x - X_data["distance_to_nearest_city"].mean())/X_data["distance_to_nearest_city"].std()

# use prediction function with the standardized input parameters given through the user
ypred = np.array([[standardise_0(mean_age), standardise_1(total_pop), standardise_2(dist_to_city), month_lst[0], month_lst[1], month_lst[2], month_lst[3], month_lst[4], month_lst[5], month_lst[6], month_lst[7], month_lst[8], month_lst[9], month_lst[10], month_lst[11], day_lst[0], day_lst[1], day_lst[2], day_lst[3], day_lst[4], day_lst[5], day_lst[6]]])
pred = predict_packages(X, y, ypred)
pred = relu(pred)

# execution of calculation button -> displaying the result of the prediction
if go:
    st.write(f"### Expected number of packages: {pred[0]:.0f}")