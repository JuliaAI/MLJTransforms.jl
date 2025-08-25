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
using Random, CSV, Plots

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

results = DataFrame(
    pipeline = String[],
    accuracy = Float64[],
    std_error = Float64[],
    ci_lower = Float64[],
    ci_upper = Float64[],
)

for (name, pipe) in pipelines
    println("Evaluating: $name")
    eval_results = evaluate(
        pipe,
        X,
        y,
        resampling = CV(nfolds = 5, rng = 123),
        measure = accuracy,
        rows = train,
        verbosity = 0,
    )
    acc = eval_results.measurement[1]          # scalar mean
    per_fold = eval_results.per_fold[1]         # vector of fold results
    se = std(per_fold) / sqrt(length(per_fold))
    ci = 1.96 * se
    push!(
        results,
        (
            pipeline = name,
            accuracy = acc,
            std_error = se,
            ci_lower = acc - ci,
            ci_upper = acc + ci,
        ),
    )
    println("  Mean accuracy: $(round(acc, digits=4)) ± $(round(ci, digits=4))")
end

# Sort results by accuracy (highest first) and display:
sort!(results, :accuracy, rev = true)

# Display results with confidence intervals
println("\nResults with 95% Confidence Intervals (see caveats below):")
println("="^60)
for row in eachrow(results)
    pipeline = row.pipeline
    acc = round(row.accuracy, digits = 4)
    ci_lower = round(row.ci_lower, digits = 4)
    ci_upper = round(row.ci_upper, digits = 4)
    println("$pipeline: $acc (95% CI: [$ci_lower, $ci_upper])")
end

results

# ## Results Analysis
# 
# ### Performance Summary
# The results show OneHot encoding performing best, followed by Target encoding, with Ordinal and Frequency encoders showing lower performance.
# 
# The confidence intervals should be interpreted with caution and primarily serve to illustrate uncertainty rather than provide definitive statistical significance tests.
# See Bengio & Grandvalet, 2004: "No Unbiased Estimator of the Variance of K-Fold Cross-Validation"). That said, reporting the interval is still more informative than reporting only the mean.

# Prepare data for plotting
labels = results.pipeline
mean_acc = results.accuracy
ci_lower = results.ci_lower
ci_upper = results.ci_upper

# Error bars: distance from mean to CI bounds
lower_err = mean_acc .- ci_lower
upper_err = ci_upper .- mean_acc

bar(
    labels,
    mean_acc,
    yerror = (lower_err, upper_err),
    legend = false,
    xlabel = "Encoder + SVM",
    ylabel = "Accuracy",
    title = "Mean Accuracy with 95% Confidence Intervals",
    ylim = (0, 1.05),
    color = :skyblue,
    size = (700, 400),
);

# save the figure and load it
savefig("encoder_comparison.png");
# ![`encoder_comparison.png`](encoder_comparison.png)
