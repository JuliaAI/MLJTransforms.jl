# # Adult Income Prediction: Comparing Categorical Encoders

# **Julia version** is assumed to be 1.10.*

# This demonstration is available as a Jupyter notebook or julia script (as well as the dataset)
# [here](https://github.com/essamwise/MLJTransforms.jl/tree/main/docs/src/tutorials/wine_example).
#
# This tutorial compares different categorical encoding approaches on adult income prediction.
# We'll test OneHot, Frequency, and Cardinality Reduction encoders with CatBoost classification.
# 
# **Why compare encoders?** Categorical variables with many levels (like occupation, education)
# can create high-dimensional sparse features. Different encoding strategies handle this
# challenge differently, affecting both model performance and training speed.
# 
# **High Cardinality Challenge:** We've added a synthetic feature with 100 categories to 
# demonstrate how encoders handle extreme cardinality - a common real-world scenario with
# features like customer IDs, product codes, or geographical subdivisions.

# packages are already activated by generate.jl

using MLJ, MLJTransforms, DataFrames, ScientificTypes
using Random, CSV, StatsBase, Plots, BenchmarkTools

# Import scitypes from MLJ to avoid any package version skew
using MLJ: OrderedFactor, Continuous, Multiclass

# ## Load and Prepare Data
# Load the Adult Income dataset. This dataset contains demographic information
# and the task is to predict whether a person makes over $50K per year.

# Load data with header and rename columns to the expected symbols
df = CSV.read("./adult.csv", DataFrame; header = true)
rename!(
    df,
    [
        :age,
        :workclass,
        :fnlwgt,
        :education,
        :education_num,
        :marital_status,
        :occupation,
        :relationship,
        :race,
        :sex,
        :capital_gain,
        :capital_loss,
        :hours_per_week,
        :native_country,
        :income,
    ],
)

first(df, 5)


# Clean the data by removing leading/trailing spaces and converting income to binary:
for col in [:workclass, :education, :marital_status, :occupation, :relationship,
    :race, :sex, :native_country, :income]
    df[!, col] = strip.(string.(df[!, col]))
end

# Convert income to binary (0 for <=50K, 1 for >50K)
df.income = ifelse.(df.income .== ">50K", 1, 0);

# Let's a high-cardinality categorical feature to showcase encoder handling
# Create a realistic frequency distribution: A1-A3 make up 90% of data, A4-A500 make up 10%
Random.seed!(42)
high_card_categories = ["A$i" for i in 1:500]

n_rows = nrow(df)
n_frequent = Int(round(0.9 * n_rows))  # 90% for A1, A2, A3
n_rare = n_rows - n_frequent           # 10% for A4-A500

frequent_samples = rand(["A1", "A2", "A3"], n_frequent)

rare_categories = ["A$i" for i in 4:500]
rare_samples = rand(rare_categories, n_rare);

# Combine and shuffle
all_samples = vcat(frequent_samples, rare_samples)
df.high_cardinality_feature = all_samples[randperm(n_rows)];

# Coerce categorical columns to appropriate scientific types. 
# Apply explicit type coercions using fully qualified names
type_dict = Dict(
    :income => OrderedFactor,
    :age => Continuous,
    :fnlwgt => Continuous,
    :education_num => Continuous,
    :capital_gain => Continuous,
    :capital_loss => Continuous,
    :hours_per_week => Continuous,
    :workclass => Multiclass,
    :education => Multiclass,
    :marital_status => Multiclass,
    :occupation => Multiclass,
    :relationship => Multiclass,
    :race => Multiclass,
    :sex => Multiclass,
    :native_country => Multiclass,
    :high_cardinality_feature => Multiclass,
)
df = coerce(df, type_dict);

# Let's examine the cardinality of our categorical features:
categorical_cols = [:workclass, :education, :marital_status, :occupation,
    :relationship, :race, :sex, :native_country, :high_cardinality_feature]
println("Cardinality of categorical features:")
for col in categorical_cols
    n_unique = length(unique(df[!, col]))
    println("  $col: $n_unique unique values")
end



# ## Split Data
# Separate features (X) from target (y), then split into train/test sets:

y, X = unpack(df, ==(:income); rng = 123);
train, test = partition(eachindex(y), 0.8, shuffle = true, rng = 100);

# ## Setup Encoders and Model
# Load the required models and create different encoding strategies:

OneHot = @load OneHotEncoder pkg = MLJModels verbosity = 0
CatBoostClassifier = @load CatBoostClassifier pkg = CatBoost


# **Encoding Strategies:**
# 1. **OneHotEncoder**: Creates binary columns for each category
# 2. **FrequencyEncoder**: Replaces categories with their frequency counts
# In case of the one-hot-encoder, we worry when categories have high cardinality as that would lead to an explosion in the number of features.

card_reducer = MLJTransforms.CardinalityReducer(
    min_frequency = 0.15,
    ordered_factor = true,
    label_for_infrequent = Dict(
        AbstractString => "OtherItems",
        Char => 'O',
    ),
)
onehot_model = OneHot(drop_last = true, ordered_factor = true)
freq_model = MLJTransforms.FrequencyEncoder(normalize = false, ordered_factor = true)
cat = CatBoostClassifier();

# Create three different pipelines to compare:
pipelines = [
    ("CardRed + OneHot + CAT", card_reducer |> onehot_model |> cat),
    ("OneHot + CAT", onehot_model |> cat),
    ("FreqEnc + CAT", freq_model |> cat),
]

# ## Evaluate Pipelines with Proper Benchmarking
# Train each pipeline and measure both performance (accuracy) and training time using @btime:

results = DataFrame(pipeline = String[], accuracy = Float64[], training_time = Float64[]);

# Prepare results DataFrame

for (name, pipe) in pipelines
    println("Training and benchmarking: $name")

    ## Train once to compute accuracy
    mach = machine(pipe, X, y)
    MLJ.fit!(mach, rows = train)
    predictions = MLJ.predict_mode(mach, rows = test)
    accuracy_value = MLJ.accuracy(predictions, y[test])

    ## Measure training time using @belapsed (returns Float64 seconds) with 5 samples
    ## Create a fresh machine inside the benchmark to avoid state sharing
    training_time =
        @belapsed MLJ.fit!(machine($pipe, $X, $y), rows = $train, force = true) samples = 5

    println("  Training time (min over 5 samples): $(training_time) s")
    println("  Accuracy: $(round(accuracy_value, digits=4))\n")

    push!(results, (string(name), accuracy_value, training_time))
end


# Sort by accuracy (higher is better) and display results:
sort!(results, :accuracy, rev = true)
results

# ## Visualization
# Create side-by-side bar charts to compare both training time and model performance:

n = nrow(results)

# Create a simple timing visualization (note: timing strings from @btime need manual parsing for plotting)
# Sort by accuracy (higher is better)
sort!(results, :accuracy, rev = true)
results  # show table

# -------------------------
# Visualization (side-by-side)
# -------------------------
n = nrow(results)
# training time plot (seconds)
time_plot = bar(1:n, results.training_time;
    xticks = (1:n, results.pipeline),
    title = "Training Time (s)",
    xlabel = "Pipeline", ylabel = "Time (s)",
    xrotation = 8,
    legend = false,
    color = :lightblue,
);

# accuracy plot
accuracy_plot = bar(1:n, results.accuracy;
    xticks = (1:n, results.pipeline),
    title = "Classification Accuracy",
    xlabel = "Pipeline", ylabel = "Accuracy",
    xrotation = 8,
    legend = false,
    ylim = (0.0, 1.0),
    color = :lightcoral,
);


combined_plot = plot(time_plot, accuracy_plot; layout = (1, 2), size = (1200, 500));

# Save the plot
savefig(combined_plot, "adult_encoding_comparison.png"); #hide

#md # ![Adult Encoding Comparison](adult_encoding_comparison.png)

# ## Conclusion
#
# **Key Findings from Results:**
# 
# **Training Time Performance (dramatic differences!):**
# - **FreqEnc + CAT**: 0.32 seconds - **fastest approach**
# - **CardRed + OneHot + CAT**: 0.57 seconds - **10x faster than pure OneHot**
# - **OneHot + CAT**: 5.85 seconds - **significantly slower due to high cardinality**
#
# **Accuracy:** In this example, we don't see a difference in accuracy but the savings in time are big.

# Note that we still observe a speed improvement with the cardinality reducer if we omit the high cardinality feature we added but it's much smaller as the adults dataset is not that high in cardinality.