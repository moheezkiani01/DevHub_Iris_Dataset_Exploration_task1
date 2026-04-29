# Exploring and Visualizing a Simple Dataset

## 🌐 Live Deployment
Available soon

## 📊 Overview
This task focuses on fundamental data exploration and visualization techniques using the classic Iris dataset. It demonstrates essential pandas operations and matplotlib/seaborn visualizations.

## 🎯 Objective
Learn how to load, inspect, and visualize a dataset to understand data trends and distributions.

## 📁 Dataset
**Iris Dataset**
- **Source:** UCI Machine Learning Repository (available via sklearn)
- **Size:** 150 samples, 4 features
- **Classes:** 3 iris species (Setosa, Versicolor, Virginica)
- **Features:**
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Plotting
- **seaborn** - Statistical visualizations
- **sklearn** - Dataset loading

## 📋 Requirements Checklist

### What This Notebook Includes:
- ✅ Load dataset using pandas
- ✅ Print shape, column names, first rows (`.head()`)
- ✅ Use `.info()` and `.describe()` for summary statistics
- ✅ Create scatter plots (feature relationships)
- ✅ Use histograms (value distributions)
- ✅ Use box plots (identify outliers)
- ✅ Use matplotlib and seaborn for plotting

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Notebook
```bash
jupyter notebook Task1_Iris_Dataset_Exploration.ipynb
```

### 3. Execute All Cells
Run all cells sequentially from top to bottom (Cell → Run All)

## 📈 Key Outputs

### Visualizations Generated:
1. **Pairplot** - Scatter plot matrix showing all feature relationships
2. **Individual Scatter Plot** - Petal Length vs Petal Width
3. **Histograms** - Distribution of each feature by species
4. **Box Plots** - Feature distributions to identify outliers
5. **Correlation Heatmap** - Feature relationships
6. **Violin Plots** - Distribution density by species

### Key Findings:
- 📊 Dataset contains 150 samples, balanced across 3 species
- 🌸 Setosa species is clearly separable from others
- 📏 Petal measurements are more distinctive than sepal measurements
- 🔗 High correlation between Petal Length and Petal Width (0.96)
- 🎯 Minimal outliers detected in the dataset

## 💡 Skills Demonstrated
- Data loading and inspection using pandas
- Descriptive statistics and data exploration
- Basic plotting and visualization with matplotlib/seaborn
- Pattern recognition and data interpretation
- Outlier detection and correlation analysis

## 📊 Notebook Structure
1. Import Libraries
2. Load Dataset
3. Data Inspection (shape, columns, info, describe)
4. Descriptive Statistics
5. Data Visualization
   - Scatter Plots
   - Histograms
   - Box Plots
   - Correlation Heatmap
   - Violin Plots
6. Key Findings and Insights
7. Summary

## 🎓 Learning Outcomes
After completing this task, you will understand:
- How to load and inspect datasets
- How to generate descriptive statistics
- How to create various types of plots
- How to identify patterns and outliers
- How to interpret correlation matrices

## 📝 Notes
- This is an exploratory data analysis (EDA) task - no machine learning models are built
- All visualizations are generated inline in the notebook
- The Iris dataset is built into sklearn, so no external download is needed

## 🔗 Additional Resources
- [Iris Dataset Info](https://archive.ics.uci.edu/ml/datasets/iris)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Status:** ✅ Complete  
**Estimated Time:** 30-45 minutes  
**Difficulty:** Beginner
