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
In `MLJTransforms` we define "encoders" to encompass models that specifically operate by encoding categorical variables; meanwhile, "transformers" refers to models that apply more generic transformations on columns that are not necessarily categorical. We define the following taxonomy for different models founds in `MLJTransforms`:

| Genre | Definition | 
|:----------:|:----------:|
| **Classical Encoders** | Well known and commonly used categorical encoders | 
| **Neural-based Encoders** | Categorical encoders based on neural networks | 
| **Contrast Encoders** | Categorical encoders that could be modeled by a contrast matrix  | 
| **Utility Encoders** | Categorical encoders meant to be used as preprocessors for other encoders or models | 
| **Other Transformers** | More generic transformers that go beyond categorical encoding   | 




