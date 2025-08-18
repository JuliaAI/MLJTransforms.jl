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
using Random, CSV
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
<div><div style = "float: left;"><span>5×8 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">pH</th><th style = "text-align: left;">Temprature</th><th style = "text-align: left;">Taste</th><th style = "text-align: left;">Odor</th><th style = "text-align: left;">Fat </th><th style = "text-align: left;">Turbidity</th><th style = "text-align: left;">Colour</th><th style = "text-align: left;">Grade</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "InlineStrings.String7" style = "text-align: left;">String7</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">6.6</td><td style = "text-align: right;">35</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">254</td><td style = "text-align: left;">high</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">6.6</td><td style = "text-align: right;">36</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">253</td><td style = "text-align: left;">high</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">8.5</td><td style = "text-align: right;">70</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">246</td><td style = "text-align: left;">low</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">9.5</td><td style = "text-align: right;">34</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">0</td><td style = "text-align: right;">1</td><td style = "text-align: right;">255</td><td style = "text-align: left;">low</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">6.6</td><td style = "text-align: right;">37</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">0</td><td style = "text-align: right;">255</td><td style = "text-align: left;">medium</td></tr></tbody></table></div>
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
````

````
Evaluating: OneHot + SVM
Evaluating: FreqEnc + SVM
Evaluating: TargetEnc + SVM
Evaluating: Ordinal + SVM

````

Sort results by accuracy (highest first) and display:

````julia
sort!(results, :accuracy, rev = true)
results
````

```@raw html
<div><div style = "float: left;"><span>4×2 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">pipeline</th><th style = "text-align: left;">accuracy</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "String" style = "text-align: left;">String</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">OneHot + SVM</td><td style = "text-align: right;">0.998951</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">TargetEnc + SVM</td><td style = "text-align: right;">0.974816</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">Ordinal + SVM</td><td style = "text-align: right;">0.940189</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">FreqEnc + SVM</td><td style = "text-align: right;">0.885624</td></tr></tbody></table></div>
```

## Results Analysis
We notice that one-hot-encoding was the most performant here followed by target encoding.
Ordinal encoding also produced decent results because we can perceive all the categorical variables to be ordered
On the other hand, frequency encoding lagged behind. Observe that this method doesn't distinguish categories from one another if they occur with similar frequencies.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

