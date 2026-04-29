import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set up the page configuration
st.set_page_config(
    page_title="🌸 Iris Dataset Explorer",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("🌸 Iris Dataset Explorer")
st.markdown("""
Explore the classic Iris dataset with interactive visualizations and machine learning predictions.
The Iris dataset is a famous dataset in machine learning that contains measurements of iris flowers.
""")

# Load the Iris dataset
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris.target_names, iris

df, target_names, iris_data = load_iris_data()

# Dataset overview
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    st.metric("Features", len(iris_data.feature_names))
with col3:
    st.metric("Classes", len(target_names))

st.write("Dataset Shape:", df.shape)
st.write("Feature Names:", iris_data.feature_names)
st.write("Target Classes:", target_names.tolist())

# Show raw data
st.subheader("Raw Data")
st.dataframe(df.head(10))

# Sidebar for visualization options
st.sidebar.header("Visualization Settings")
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap", "Pair Plot"]
)
x_feature = st.sidebar.selectbox(
    "X-axis Feature",
    options=iris_data.feature_names,
    index=0
)
y_feature = st.sidebar.selectbox(
    "Y-axis Feature",
    options=iris_data.feature_names,
    index=1
)

# Visualization section
st.subheader(f"{plot_type} Visualization")

if plot_type == "Scatter Plot":
    fig, ax = plt.subplots(figsize=(10, 6))
    for species in df['species_name'].unique():
        subset = df[df['species_name'] == species]
        ax.scatter(subset[x_feature], subset[y_feature], label=species, alpha=0.7, s=60)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f'{y_feature} vs {x_feature} by Species')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

elif plot_type == "Histogram":
    fig, ax = plt.subplots(figsize=(10, 6))
    for species in df['species_name'].unique():
        subset = df[df['species_name'] == species]
        ax.hist(subset[x_feature], alpha=0.6, label=species, bins=15, edgecolor='black')
    ax.set_xlabel(x_feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {x_feature} by Species')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

elif plot_type == "Box Plot":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='species_name', y=x_feature, ax=ax)
    ax.set_xlabel('Species')
    ax.set_ylabel(x_feature)
    ax.set_title(f'Box Plot of {x_feature} by Species')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif plot_type == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[iris_data.feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)

elif plot_type == "Pair Plot":
    # Create a smaller pairplot for performance
    sample_df = df.sample(min(150, len(df)))  # Take a sample for faster plotting
    fig = sns.pairplot(sample_df, hue='species_name', vars=iris_data.feature_names,
                       height=2.5, aspect=1.2)
    fig.fig.suptitle('Pair Plot of All Features', y=1.02)
    st.pyplot(fig)

# Species distribution
st.subheader("Species Distribution")
species_counts = df['species_name'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
ax.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title('Distribution of Species in Dataset')
st.pyplot(fig)

# Statistical Summary
st.subheader("Statistical Summary")
st.write(df.describe())

# Machine Learning Model
st.subheader("Machine Learning Model")
st.write("Training a Random Forest classifier to predict iris species based on features...")

# Prepare data for modeling
X = df[iris_data.feature_names]
y = df['species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display results
col1, col2 = st.columns(2)
with col1:
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
with col2:
    st.metric("Test Samples", len(X_test))

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': iris_data.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['Feature'], feature_importance['Importance'])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance for Species Classification')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
st.pyplot(fig)

# Interactive prediction
st.subheader("Interactive Prediction")
st.write("Enter feature values to predict the iris species:")

# Input fields for prediction
col1, col2, col3, col4 = st.columns(4)
with col1:
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
with col2:
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=5.0, value=3.0)
with col3:
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
with col4:
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=5.0, value=1.0)

# Make prediction
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.success(f"Predicted Species: {target_names[prediction]}")
    
    # Show prediction probabilities
    prob_df = pd.DataFrame({
        'Species': target_names,
        'Probability': prediction_proba
    }).sort_values('Probability', ascending=False)
    
    st.write("Prediction Probabilities:")
    st.dataframe(prob_df)

# Dataset Information
st.subheader("About the Iris Dataset")
st.write("""
The Iris dataset contains 3 classes of flowers:
- **Setosa**: Known for its distinctive shape and color
- **Versicolor**: A versicolored flower species
- **Virginica**: A more elongated flower species

**Features measured:**
- Sepal length and width (in cm)
- Petal length and width (in cm)

This dataset is commonly used as an introductory dataset for machine learning classification tasks.
""")

# Footer
st.markdown("---")
st.markdown("Developed as part of AI/ML Engineering Internship Tasks")
st.markdown("This application explores the classic Iris dataset with interactive visualizations and machine learning.")