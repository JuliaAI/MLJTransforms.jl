```@meta
EditURL = "notebook.jl"
```

# Categorical Encoders Performance: A Classic Comparison

**Julia version** is assumed to be 1.10.*

This demonstration is available as a Jupyter notebook or julia script (as well as the dataset)
[here](https://github.com/essamwise/MLJTransforms.jl/tree/main/docs/src/tutorials/classic_comparison).

This tutorial compares four fundamental categorical encoding approaches on a milk quality dataset:
OneHot, Frequency, Target, and Ordinal encoders paired with SVM classification.

````julia
using Pkg;
Pkg.activate(@__DIR__);

using MLJ, MLJTransforms, LIBSVM, DataFrames, ScientificTypes
using Random, CSV, Plots
````

````
  Activating project at `~/Documents/GitHub/MLJTransforms/docs/src/tutorials/classic_comparison`

````

## Load and Prepare Data
Load the milk quality dataset which contains categorical features for quality prediction:

````julia
df = CSV.read("./milknew.csv", DataFrame)

first(df, 5)
````

```@raw html
<div><div style = "float: left;"><span>5×8 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">pH</th><th style = "text-align: left;">Temprature</th><th style = "text-align: left;">Taste</th><th style = "text-align: left;">Odor</th><th style = "text-align: left;">Fat </th><th style = "text-align: left;">Turbidity</th><th style = "text-align: left;">Colour</th><th style = "text-align: left;">Grade</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "String7" style = "text-align: left;">String7</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">6.6</td><td style = "text-align: right;">35</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">254</td><td style = "text-align: left;">high</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">6.6</td><td style = "text-align: right;">36</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">253</td><td style = "text-align: left;">high</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">8.5</td><td style = "text-align: right;">70</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">246</td><td style = "text-align: left;">low</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">9.5</td><td style = "text-align: right;">34</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">255</td><td style = "text-align: left;">low</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">6.6</td><td style = "text-align: right;">37</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">255</td><td style = "text-align: left;">medium</td></tr></tbody></table></div>
```

Check the scientific types to understand our data structure:

````julia
ScientificTypes.schema(df)
````

````
┌────────────┬────────────┬─────────┐
│ names      │ scitypes   │ types   │
├────────────┼────────────┼─────────┤
│ pH         │ Continuous │ Float64 │
│ Temprature │ Count      │ Int64   │
│ Taste      │ Count      │ Int64   │
│ Odor       │ Count      │ Int64   │
│ Fat        │ Count      │ Int64   │
│ Turbidity  │ Count      │ Int64   │
│ Colour     │ Count      │ Int64   │
│ Grade      │ Textual    │ String7 │
└────────────┴────────────┴─────────┘

````

Automatically coerce columns with few unique values to categorical:

````julia
df = coerce(df, autotype(df, :few_to_finite))

ScientificTypes.schema(df)
````

````
┌────────────┬───────────────────┬───────────────────────────────────┐
│ names      │ scitypes          │ types                             │
├────────────┼───────────────────┼───────────────────────────────────┤
│ pH         │ OrderedFactor{16} │ CategoricalValue{Float64, UInt32} │
│ Temprature │ OrderedFactor{17} │ CategoricalValue{Int64, UInt32}   │
│ Taste      │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ Odor       │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ Fat        │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ Turbidity  │ OrderedFactor{2}  │ CategoricalValue{Int64, UInt32}   │
│ Colour     │ OrderedFactor{9}  │ CategoricalValue{Int64, UInt32}   │
│ Grade      │ Multiclass{3}     │ CategoricalValue{String7, UInt32} │
└────────────┴───────────────────┴───────────────────────────────────┘

````

## Split Data
Separate features from target and create train/test split:

````julia
y, X = unpack(df, ==(:Grade); rng = 123)
train, test = partition(eachindex(y), 0.9, shuffle = true, rng = 100);
````

## Setup Encoders and Classifier
Load the required models and create different encoding strategies:

````julia
OneHot = @load OneHotEncoder pkg = MLJModels verbosity = 0
SVC = @load SVC pkg = LIBSVM verbosity = 0
````

````
MLJLIBSVMInterface.SVC
````

**Encoding Strategies Explained:**
1. **OneHot**: Creates binary columns for each category (sparse, interpretable)
2. **Frequency**: Replaces categories with their occurrence frequency
3. **Target**: Uses target statistics for each category
4. **Ordinal**: Assigns integer codes to categories (assumes ordering)

````julia
onehot_model = OneHot(drop_last = true, ordered_factor = true)
freq_model = MLJTransforms.FrequencyEncoder(normalize = false, ordered_factor = true)
target_model = MLJTransforms.TargetEncoder(lambda = 0.9, m = 5, ordered_factor = true)
ordinal_model = MLJTransforms.OrdinalEncoder(ordered_factor = true)
svm = SVC()
````

````
SVC(
  kernel = LIBSVM.Kernel.RadialBasis, 
  gamma = 0.0, 
  cost = 1.0, 
  cachesize = 200.0, 
  degree = 3, 
  coef0 = 0.0, 
  tolerance = 0.001, 
  shrinking = true)
````

Create four different pipelines to compare:

````julia
pipelines = [
    ("OneHot + SVM", onehot_model |> svm),
    ("FreqEnc + SVM", freq_model |> svm),
    ("TargetEnc + SVM", target_model |> svm),
    ("Ordinal + SVM", ordinal_model |> svm),
]
````

````
4-element Vector{Tuple{String, MLJBase.DeterministicPipeline{N, MLJModelInterface.predict} where N<:NamedTuple}}:
 ("OneHot + SVM", DeterministicPipeline(one_hot_encoder = OneHotEncoder(features = Symbol[], …), …))
 ("FreqEnc + SVM", DeterministicPipeline(frequency_encoder = FrequencyEncoder(features = Symbol[], …), …))
 ("TargetEnc + SVM", DeterministicPipeline(target_encoder = TargetEncoder(features = Symbol[], …), …))
 ("Ordinal + SVM", DeterministicPipeline(ordinal_encoder = OrdinalEncoder(features = Symbol[], …), …))
````

## Evaluate Pipelines
Use 10-fold cross-validation to robustly estimate each pipeline's accuracy:

````julia
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
````

````
Evaluating: OneHot + SVM
  Mean accuracy: 0.999 ± 0.0021
Evaluating: FreqEnc + SVM
  Mean accuracy: 0.8804 ± 0.0286
Evaluating: TargetEnc + SVM
  Mean accuracy: 0.9738 ± 0.0086
Evaluating: Ordinal + SVM
  Mean accuracy: 0.9328 ± 0.0119

````

Sort results by accuracy (highest first) and display:

````julia
sort!(results, :accuracy, rev = true)
````

```@raw html
<div><div style = "float: left;"><span>4×5 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">pipeline</th><th style = "text-align: left;">accuracy</th><th style = "text-align: left;">std_error</th><th style = "text-align: left;">ci_lower</th><th style = "text-align: left;">ci_upper</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "String" style = "text-align: left;">String</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">OneHot + SVM</td><td style = "text-align: right;">0.998951</td><td style = "text-align: right;">0.00105263</td><td style = "text-align: right;">0.996888</td><td style = "text-align: right;">1.00101</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">TargetEnc + SVM</td><td style = "text-align: right;">0.973767</td><td style = "text-align: right;">0.00441017</td><td style = "text-align: right;">0.965123</td><td style = "text-align: right;">0.982411</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">Ordinal + SVM</td><td style = "text-align: right;">0.932844</td><td style = "text-align: right;">0.00606985</td><td style = "text-align: right;">0.920947</td><td style = "text-align: right;">0.944741</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">FreqEnc + SVM</td><td style = "text-align: right;">0.880378</td><td style = "text-align: right;">0.0145961</td><td style = "text-align: right;">0.851769</td><td style = "text-align: right;">0.908986</td></tr></tbody></table></div>
```

Display results with confidence intervals

````julia
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
````

```@raw html
<div><div style = "float: left;"><span>4×5 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">pipeline</th><th style = "text-align: left;">accuracy</th><th style = "text-align: left;">std_error</th><th style = "text-align: left;">ci_lower</th><th style = "text-align: left;">ci_upper</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "String" style = "text-align: left;">String</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">OneHot + SVM</td><td style = "text-align: right;">0.998951</td><td style = "text-align: right;">0.00105263</td><td style = "text-align: right;">0.996888</td><td style = "text-align: right;">1.00101</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">TargetEnc + SVM</td><td style = "text-align: right;">0.973767</td><td style = "text-align: right;">0.00441017</td><td style = "text-align: right;">0.965123</td><td style = "text-align: right;">0.982411</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">Ordinal + SVM</td><td style = "text-align: right;">0.932844</td><td style = "text-align: right;">0.00606985</td><td style = "text-align: right;">0.920947</td><td style = "text-align: right;">0.944741</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">FreqEnc + SVM</td><td style = "text-align: right;">0.880378</td><td style = "text-align: right;">0.0145961</td><td style = "text-align: right;">0.851769</td><td style = "text-align: right;">0.908986</td></tr></tbody></table></div>
```

## Results Analysis

### Performance Summary
The results show OneHot encoding performing best, followed by Target encoding, with Ordinal and Frequency encoders showing lower performance.

The confidence intervals should be interpreted with caution and primarily serve to illustrate uncertainty rather than provide definitive statistical significance tests.
See Bengio & Grandvalet, 2004: "No Unbiased Estimator of the Variance of K-Fold Cross-Validation"). That said, reporting the interval is still more informative than reporting only the mean.

Prepare data for plotting

````julia
labels = results.pipeline
mean_acc = results.accuracy
ci_lower = results.ci_lower
ci_upper = results.ci_upper
````

````
4-element Vector{Float64}:
 1.0010138399514
 0.9824109872813186
 0.9447405610093282
 0.9089860558215551
````

Error bars: distance from mean to CI bounds

````julia
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
````

save the figure and load it

````julia
savefig("encoder_comparison.png");
````

![`encoder_comparison.png`](encoder_comparison.png)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

