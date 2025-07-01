# MLJTransforms.jl

A Julia package providing a wide range of categorical encoders and transformers to be used with the [MLJ](https://juliaai.github.io/MLJ.jl/dev/) package. Transformers help convert raw features into a representation that's better suited for downstream models. Meanwhile, categorical encoders are a type of transformer that specifically encodes categorical features into numerical forms. 

## Installation

```julia
import Pkg
Pkg.activate("my_environment", shared=true)
Pkg.add(["MLJ", "MLJTransforms"])
```

## Quick Start

For the following demo, you will need to additionally run `Pkg.add("RDatasets")`.

```julia
using MLJ, MLJTransforms
import RDatasets

# 1. Load Data
X = RDatasets.dataset("HSAUR", "Forbes2000");

# 2. Load the model
FrequencyEncoder = @load FrequencyEncoder pkg="MLJTransforms"
encoder = FrequencyEncoder(
    features=[:Country, :Category],     # The categorical columns to select
    ignore=false,                       # Whether to exclude or include selected columns
    ordered_factor = false,             # Whether to also encode columns of ordered factor elements
    normalize=true                      # Whether to normalize the frequencies used for encoding
    )


# 3. Wrap it in a machine and fit
mach = fit!(machine(encoder, X))
Xnew = transform(mach, X)
```

## Available Transformers
See [complete list](transformers/all_transformers) of transformers in this package.

In `MLJTransforms` we denote transformers that can operate on columns with `Continuous` and/or `Count` [scientific types](https://juliaai.github.io/ScientificTypes.jl/dev/) as *numerical transformers*. Meanwhile, *categorical transformers* operate on `Multiclass` and/or `OrderedFactor` [scientific types](https://juliaai.github.io/ScientificTypes.jl/dev/). Most categorical transformers in this package operate by converting categorical values into numerical values or vectors, and are therefore considered categorical encoders. We categorize categorical encoders as follows:


| **Category**                | **Description**                                                                 |
|:---------------------------:|:-------------------------------------------------------------------------------:|
| [Classical Encoders](transformers/classical.md)       | Traditional categorical encoding algorithms and techniques.                 |
| [Neural-based Encoders](transformers/neural)    | Categorical encoders based on neural networks.                                  |
| [Contrast Encoders](transformers/contrast.md)        | Categorical encoders that could be modeled via a contrast matrix.                             |
| [Utility Encoders](transformers/utility.md)         | Categorical encoders meant to be used as preprocessors for other transformers or models.|


Some transformers in this package can even operate on both `Finite` and `Infinite` scientific types or other special scientific types (eg, to represent time). To learn more about scientific types see [the official documentation](https://juliaai.github.io/ScientificTypes.jl/dev/#Type-hierarchy).