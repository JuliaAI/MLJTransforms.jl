Contrast Encoders include categorical encoders that could be modeled by a contrast matrix:

| Transformer | Brief Description | 
|:----------:|:----------:|
| [DummyEncoder](@ref) | Encodes by comparing each level to the reference level, intercept being the cell mean of the reference group | 
| [SumEncoder](@ref) | Encodes by comparing each level to the reference level, intercept being the grand mean | 
| [HelmertEncoder](@ref) | Encodes by comparing levels of a variable with the mean of the subsequent levels of the variable
| 
| [HelmertEncoder](@ref) | Encodes by comparing levels of a variable with the mean of the subsequent levels of the variable | 
| [ForwardDifferenceEncoder](@ref) | Encodes by comparing adjacent levels of a variable (each level minus the next level)
| 
| [ContrastEncoder](@ref) | Allows defining a custom contrast encoder via a contrast matrix | 
| [HypothesisEncoder](@ref) | Allows defining a custom contrast encoder via a hypothesis matrix | 


```@docs
MLJTransforms.ContrastEncoder
```