# # Effects of Feature Standardization on Model Performance
#
# Welcome to this tutorial on feature standardization in machine learning!
# In this tutorial, we'll explore how standardizing features can significantly 
# impact the performance of different machine learning models.
#
# We'll compare Logistic Regression and Support Vector Machine (SVM) models,
# both with and without feature standardization. This will help us understand
# when and why preprocessing is important for model performance.

# This demonstration is available as a Jupyter notebook or julia script
# [here](https://github.com/essamwise/MLJTransforms.jl/tree/main/docs/src/tutorials/standardization).
#

using Pkg     #md
Pkg.activate(@__DIR__);     #md
Pkg.instantiate();     #md

# ## Setup
#
# First, let's make sure we're using a compatible Julia version. This code was tested with Julia 1.10.
# Let's import all the packages we'll need for this tutorial.

## Load the necessary packages
using MLJ                   # Core MLJ framework
using LIBSVM                # For Support Vector Machine
using DataFrames            # For displaying results
using RDatasets             # To load sample datasets
using Random                # For reproducibility
using ScientificTypes       # For proper data typing
using Plots                 # For visualizations
using MLJLinearModels       # For Logistic Regression

# ## Data Preparation
#
# Let's load the Pima Indians Diabetes Dataset. This is a classic dataset for
# binary classification, where we predict diabetes status based on various health metrics.
# 
# The interesting thing about this dataset is that different features have very different scales.
# We'll artificially exaggerate this by adding a large constant to the glucose values.

## Load the dataset and modify it to have extreme scale differences
df = RDatasets.dataset("MASS", "Pima.tr")
df.Glu .+= 10000.0;  # Artificially increase the scale of glucose values

# Let's examine the first few rows of our dataset:
first(df, 5)

# ### Data Type Conversion
#
# In MLJ, it's important to ensure that our data has the correct scientific types.
# This helps the framework understand how to properly handle each column.
#
# We'll convert our columns to their appropriate types:
# - `Count` for discrete count data
# - `Continuous` for continuous numerical data
# - `Multiclass` for our target variable

## Coerce columns to the right scientific types
df = coerce(df,
    :NPreg => Count,      # Number of pregnancies is a count
    :Glu => Continuous,   # Glucose level is continuous
    :BP => Continuous,    # Blood pressure is continuous
    :Skin => Continuous,  # Skin thickness is continuous
    :BMI => Continuous,   # Body mass index is continuous
    :Ped => Continuous,   # Diabetes pedigree is continuous
    :Age => Continuous,   # Age is continuous
    :Type => Multiclass,  # Diabetes status is our target (Yes/No)
);

# Let's verify that our schema looks correct:
ScientificTypes.schema(df)

# ## Feature Extraction and Data Splitting
#
# Now we'll separate our features from our target variable.
# In MLJ, this is done with the `unpack` function.

## Unpack features (X) and target (y)
y, X = unpack(df, ==(:Type); rng = 123);

# Next, we'll split our data into training and testing sets.
# We'll use 70% for training and 30% for testing.

## Split data into train and test sets
train, test = partition(eachindex(y), 0.7, shuffle = true, rng = 123);

# ## Model Setup
#
# We'll compare two different models:
# 1. Logistic Regression: A linear model good for binary classification
# 2. Support Vector Machine (SVM): A powerful non-linear classifier
#
# For each model, we'll create two versions:
# - One without standardization
# - One with standardization
#
# The `Standardizer` transformer will rescale our features to have mean 0 and standard deviation 1.

## Load our models from their respective packages
logreg = @load LogisticClassifier pkg = MLJLinearModels
svm = @load SVC pkg = LIBSVM
stand = Standardizer()  # This is our standardization transformer

## Create pipelines for each model variant
logreg_pipe = logreg()  # Plain logistic regression
logreg_std_pipe = Pipeline(stand, logreg())  # Logistic regression with standardization
svm_pipe = svm()  # Plain SVM
svm_std_pipe = Pipeline(stand, svm())  # SVM with standardization

# ## Model Evaluation
#
# Let's set up a vector of our models so we can evaluate them all using the same process.
# For each model, we'll store its name and the corresponding pipeline.

## Create a list of models to evaluate
models = [
    ("Logistic Regression", logreg_pipe),
    ("Logistic Regression (standardized)", logreg_std_pipe),
    ("SVM", svm_pipe),
    ("SVM (standardized)", svm_std_pipe),
]

# Now we'll loop through each model, train it, make predictions, and calculate accuracy.
# This will help us compare how standardization affects each model's performance.

## Train and evaluate each model
results = DataFrame(model = String[], accuracy = Float64[])
for (name, model) in models
    ## Create a machine learning model
    mach = machine(model, X, y)

    ## Train the model on the training data
    MLJ.fit!(mach, rows = train)

    ## Make predictions on the test data
    ## Note: Logistic regression returns probabilities, so we need to get the mode
    yhat =
        occursin("Logistic Regression", name) ?
        MLJ.predict_mode(mach, rows = test) :  # Get most likely class for logistic regression
        MLJ.predict(mach, rows = test)         # SVM directly predicts the class

    ## Calculate accuracy
    acc = accuracy(yhat, y[test])

    ## Store the results
    push!(results, (name, acc))
end

# ## Results Visualization
#
# Finally, let's visualize our results to see the impact of standardization.
# We'll create a bar chart comparing the accuracy of each model.

## Create a bar chart of model performance
p = bar(
    results.model,
    results.accuracy,
    xlabel = "Model",
    ylabel = "Accuracy",
    title = "Model Accuracy Comparison",
    legend = false,
    bar_width = 0.6,
    ylims = (0.5, 0.7),
    xrotation = 17,
);

# Save the plot
savefig(p, "standardization_results.png"); #hide

#md # ![Model Accuracy Comparison](standardization_results.png)

# ## Conclusion
#
# From this tutorial, we can clearly see that standardization has a dramatic impact on model performance.
# 
# Looking at the results:
# 
# - **Logistic Regression**: Without standardization, it achieves only ~57% accuracy. With standardization,
#   its performance jumps dramatically to ~68% accuracy – the best performance among all models.
#
# - **SVM**: The baseline SVM achieves ~62% accuracy. When standardized, it improves to ~65% accuracy,
#   which is a significant boost but not as dramatic as what we see with logistic regression.
#
# Try this approach with other datasets and models to further explore the effects of standardization!
#
#md # ## Further Resources
#md #
#md # * [MLJTransforms Documentation](@__REPO_ROOT_URL__)
#md # * [Scientific Types in MLJ](https://alan-turing-institute.github.io/ScientificTypes.jl/dev/)
#md # * [Feature Preprocessing in MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/)
