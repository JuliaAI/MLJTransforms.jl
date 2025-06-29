```@meta
CollapsedDocStrings = true
```
### Adding new models to MLJTransforms

In this package, data transformers are not implemented using a specific generic template, whereas categorical encoders are due to their systematic nature of encoding categorical levels into scalars or vectors. In light of this, the most pivotal method in implementing a new categorical encoder is:
```@docs
MLJTransforms.generic_fit
```

followed by:
```@docs
MLJTransforms.generic_transform
```

All categorical encoders in this packager are implemented using these two methods. For an example, see `FrequencyEncoder` [source code](https://github.com/JuliaAI/MLJTransforms.jl/blob/docs/src/encoders/frequency_encoding/frequency_encoding.jl).

Moreover, you should implement the `MLJModelInterface` for any method you provide in this package. Check the interface [docs](https://juliaai.github.io/MLJModelInterface.jl/stable/) and/or the existing interfaces in this package (eg, [this interface](https://github.com/JuliaAI/MLJTransforms.jl/blob/docs/src/encoders/frequency_encoding/interface_mlj.jl) for the `FrequencyEncoder`).