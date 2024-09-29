# MLJTransforms.jl

A Julia package providing a wide range of categorical encoders and transformers to be used with the [MLJ](https://juliaai.github.io/MLJ.jl/dev/) package.

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
    features=[:Country, :Category], 
    ignore=false, ordered_factor = false, 
    normalize=true)


# 3. Wrap it in a machine and fit
mach = fit!(machine(encoder, X))
Xnew = transform(mach, X)
```

## Available Transformers
In `MLJTransforms` we denote transformers that operate on columns with `Continuous` and/or `Count` [scientific types](https://juliaai.github.io/ScientificTypes.jl/dev/) as numerical transformers. Meanwhile, categorical transformers operate on `Multiclass` and/or `OrderedFactor` [scientific types](https://juliaai.github.io/ScientificTypes.jl/dev/). Most categorical transformers in this package operate by converting categorical values into numerical values or vectors, and are therefore considered categorical encoders.

Based on this, we categorize the methods as follows, with further distinctions for categorical encoders:

| **Category**                | **Description**                                                                 |
|:---------------------------:|:-------------------------------------------------------------------------------:|
| **Numerical Transformers**   | Transformers that operate on `Continuous` or `Count` columns in a given dataset.|
| **Classical Encoders**       | Widely recognized and frequently utilized categorical encoders.                 |
| **Neural-based Encoders**    | Categorical encoders based on neural networks.                                  |
| **Contrast Encoders**        | Categorical encoders modeled via a contrast matrix.                             |
| **Utility Encoders**         | Categorical encoders meant to be used as preprocessors for other encoders or models.|
| **Other Transformers**       | Transformers that fall into other categories.                                   |