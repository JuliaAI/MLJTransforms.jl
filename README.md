# MLJTransforms.jl

A Julia package providing a wide range of categorical encoders and data transformers to be used with the [MLJ](https://juliaai.github.io/MLJ.jl/dev/) package.

[![Build Status](https://github.com/JuliaAI/Imbalance.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/Imbalance.jl/actions)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/MLJTransforms.jl/dev/)

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

See the full [documentation](https://juliaai.github.io/MLJTransforms.jl/dev/) for more information and a [full list](https://juliaai.github.io/MLJTransforms.jl/dev/all_transformers) of transformers in this package.

## 👥 Credits
This package was created by [Essam Wisam](https://github.com/JuliaAI) as a Google Summer of Code project on categorical encoding, under the mentorship of [Anthony Blaom](https://ablaom.github.io). Subsequently, the package was expanded to include data transformation methods beyond categorical encoders that previously existed in other packages.