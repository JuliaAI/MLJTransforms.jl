Other Transformers include more generic transformers that go beyond categorical encoding

| Transformer | Brief Description | 
|:----------:|:----------:|
| [Standardizer](@ref) | Transforming columns of numerical features by standardization | 
| [BoxCoxTransformer](@ref) | Transforming columns of numerical features by BoxCox transformation | 
| [UnivariateBoxCoxTransformer](@ref) | Apply BoxCox transformation given a single vector | 
| [InteractionTransformer](@ref) | Transforming columns of numerical features to create new interaction features |
| [UnivariateDiscretizer](@ref) | Discretize a continuous vector into an ordered factor | 
| [FillImputer](@ref) | Fill missing values of features belonging to any finite or infinite scientific type | 

```@docs
MLJTransforms.Standardizer
```

```@docs
MLJTransforms.InteractionTransformer
```

```@docs
MLJTransforms.BoxCoxTransformer
```

```@docs
MLJTransforms.UnivariateDiscretizer
```

```@docs
MLJTransforms.FillImputer
```