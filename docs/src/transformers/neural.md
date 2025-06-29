Neural-based Encoders include categorical encoders based on neural networks:

| Transformer | Brief Description |
|:----------:|:----------:|
| [EntityEmbedder](@ref) | Encode categorical variables into dense embedding vectors |

This method has been implemented in the [MLJFlux.jl](https://github.com/FluxML/MLJFlux.jl) which is a package that provides `MLJ` interfaces to deep learning models from [Flux.jl](https://fluxml.ai/Flux.jl/stable/). The `EntityEmbedder` encodes categorical features into dense vectors using a deep learning model that we must supply such as `MLJFlux.NeuralNetworkClassifier` or `MLJFlux.NeuralNetworkRegressor`. See the full docstring for more information:

```@docs
MLJFlux.EntityEmbedder
```