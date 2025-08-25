### Summary Table
| Transformer | Brief Description | 
|:----------:|:----------:|
| [Standardizer](@ref) | Transforming columns of numerical features by standardization | 
| [UnivariateBoxCoxTransformer](@ref) | Apply BoxCox transformation given a single vector | 
| [InteractionTransformer](@ref) | Transforming columns of numerical features to create new interaction features |
| [UnivariateDiscretizer](@ref) | Discretize a continuous vector into an ordered factor | 
| [FillImputer](@ref) | Fill missing values of features belonging to any scientific type | 
| [UnivariateTimeTypeToContinuous](@ref) | Transform a vector of time type into continuous type | 
| [OneHotEncoder](@ref) | Encode categorical variables into one-hot vectors | 
| [ContinuousEncoder](@ref) | Adds type casting functionality to OnehotEncoder | 
| [OrdinalEncoder](@ref) | Encode categorical variables into ordered integers | 
| [FrequencyEncoder](@ref) | Encode categorical variables into their normalized or unormalized frequencies | 
| [TargetEncoder](@ref) | Encode categorical variables into relevant target statistics | 
| [DummyEncoder](@ref) | Encodes by comparing each level to the reference level, intercept being the cell mean of the reference group | 
| [SumEncoder](@ref) | Encodes by comparing each level to the reference level, intercept being the grand mean | 
| [HelmertEncoder](@ref) | Encodes by comparing levels of a variable with the mean of the subsequent levels of the variable
| [ForwardDifferenceEncoder](@ref) | Encodes by comparing adjacent levels of a variable (each level minus the next level)
| [ContrastEncoder](@ref) | Allows defining a custom contrast encoder via a contrast matrix | 
| [HypothesisEncoder](@ref) | Allows defining a custom contrast encoder via a hypothesis matrix | 
| [EntityEmbedders](@ref) | Encode categorical variables into dense embedding vectors |
| [CardinalityReducer](@ref) | Reduce cardinality of high cardinality categorical features by grouping infrequent categories |
| [MissingnessEncoder](@ref) | Encode missing values of categorical features into new values |

### All Transformers

```@docs; canonical = false
MLJTransforms.Standardizer
```

```@docs; canonical = false
MLJTransforms.InteractionTransformer
```

```@docs; canonical = false
MLJTransforms.UnivariateDiscretizer
```

```@docs; canonical = false
MLJTransforms.FillImputer
```

```@docs; canonical = false
MLJTransforms.UnivariateTimeTypeToContinuous
```

```@docs; canonical = false
MLJTransforms.OneHotEncoder
```

```@docs; canonical = false
MLJTransforms.ContinuousEncoder
```

```@docs; canonical = false
MLJTransforms.OrdinalEncoder
```

```@docs; canonical = false
MLJTransforms.FrequencyEncoder
```

```@docs; canonical = false
MLJTransforms.TargetEncoder
```

```@docs; canonical = false
MLJTransforms.ContrastEncoder
```

```@docs; canonical = false
MLJTransforms.CardinalityReducer
```

```@docs; canonical = false
MLJTransforms.MissingnessEncoder
```