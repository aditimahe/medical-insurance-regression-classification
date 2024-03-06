# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title('Regression and Classification')

    st.subheader('Raw Data')

    # The URL of the CSV file to be read into a DataFrame
    csv_url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

    # Reading the CSV data from the specified URL into a DataFrame named 'df'
    df = pd.read_csv(csv_url)

    # Display the dataset
    st.write(df)

    # Remove duplicate row from dataset
    df.drop_duplicates(keep='first', inplace=True)

    # Encode 'sex', 'smoker', and 'region' columns
    df['sex_encode'] = LabelEncoder().fit_transform(df['sex'])
    df['smoker_encode'] = LabelEncoder().fit_transform(df['smoker'])
    df['region_encode'] = LabelEncoder().fit_transform(df['region'])


    # Transform the 'charges' variable using Box-Cox transformation
    df['charges_transform'] = stats.boxcox(df['charges'])[0]

    # Define X (features) and y (target) and remove duplicate features that will not be used in the model
    X = df.drop(['sex', 'smoker', 'region', 'charges', 'charges',
                'charges_transform'], axis=1)
    y = df['charges_transform']

    # Split the dataset into X_train, X_test, y_train, and y_test, 10% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Instantiate a linear regression model
    linear_model = LinearRegression()

    # Fit the model using the training data
    linear_model.fit(X_train, y_train)

    # For each record in the test set, predict the y value (transformed value of charges)
    # The predicted values are stored in the y_pred array
    y_pred = linear_model.predict(X_test)

    st.write(y_pred)

if __name__ == "__main__":
    run()
