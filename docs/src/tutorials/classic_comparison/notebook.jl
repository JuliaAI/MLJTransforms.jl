# # Categorical Encoders Performance: A Classic Comparison

# **Julia version** is assumed to be 1.10.*

# This demonstration is available as a Jupyter notebook or julia script (as well as the dataset)
# [here](https://github.com/essamwise/MLJTransforms.jl/tree/main/docs/src/tutorials/classic_comparison).
#
# This tutorial compares four fundamental categorical encoding approaches on a milk quality dataset:
# OneHot, Frequency, Target, and Ordinal encoders paired with SVM classification.
#

using Pkg;
Pkg.activate(@__DIR__);
Pkg.instantiate(); #src

using MLJ, MLJTransforms, LIBSVM, DataFrames, ScientificTypes
using Random, CSV

# ## Load and Prepare Data
# Load the milk quality dataset which contains categorical features for quality prediction:

df = CSV.read("./milknew.csv", DataFrame)

first(df, 5)

#-

# Check the scientific types to understand our data structure:
ScientificTypes.schema(df)

#-

# Automatically coerce columns with few unique values to categorical:
df = coerce(df, autotype(df, :few_to_finite))

ScientificTypes.schema(df)

# ## Split Data
# Separate features from target and create train/test split:
y, X = unpack(df, ==(:Grade); rng = 123)
train, test = partition(eachindex(y), 0.9, shuffle = true, rng = 100);

# ## Setup Encoders and Classifier
# Load the required models and create different encoding strategies:

OneHot = @load OneHotEncoder pkg = MLJModels verbosity = 0
SVC = @load SVC pkg = LIBSVM verbosity = 0

# **Encoding Strategies Explained:**
# 1. **OneHot**: Creates binary columns for each category (sparse, interpretable)
# 2. **Frequency**: Replaces categories with their occurrence frequency
# 3. **Target**: Uses target statistics for each category 
# 4. **Ordinal**: Assigns integer codes to categories (assumes ordering)

onehot_model = OneHot(drop_last = true, ordered_factor = true)
freq_model = MLJTransforms.FrequencyEncoder(normalize = false, ordered_factor = true)
target_model = MLJTransforms.TargetEncoder(lambda = 0.9, m = 5, ordered_factor = true)
ordinal_model = MLJTransforms.OrdinalEncoder(ordered_factor = true)
svm = SVC()

# Create four different pipelines to compare:
pipelines = [
    ("OneHot + SVM", onehot_model |> svm),
    ("FreqEnc + SVM", freq_model |> svm),
    ("TargetEnc + SVM", target_model |> svm),
    ("Ordinal + SVM", ordinal_model |> svm),
]


# ## Evaluate Pipelines
# Use 10-fold cross-validation to robustly estimate each pipeline's accuracy:

results = DataFrame(pipeline = String[], accuracy = Float64[])

for (name, pipe) in pipelines
    println("Evaluating: $name")
    mach = machine(pipe, X, y)
    eval_results = evaluate!(
        mach,
        resampling = CV(nfolds = 10, rng = 123),
        measure = accuracy,
        rows = train,
        verbosity = 0,
    )
    acc = mean(eval_results.measurement)
    push!(results, (name, acc))
end

# Sort results by accuracy (highest first) and display:
sort!(results, :accuracy, rev = true)
results

# ## Results Analysis
# We notice that one-hot-encoding was the most performant here followed by target encoding.
# Ordinal encoding also produced decent results because we can perceive all the categorical variables to be ordered
# On the other hand, frequency encoding lagged behind. Observe that this method doesn't distinguish categories from one another if they occur with similar frequencies.
#