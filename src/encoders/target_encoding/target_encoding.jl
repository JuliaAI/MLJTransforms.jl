"""
**Private method.**

Given the targets belonging to a specific category (level) for a categorical variable,
find the frequency of the reference class (binary classification is assumed).

# Arguments

  - `targets_for_level`: An abstract vector containing the target Multiclass or OrderedFactor values
  - `y_classes`: An abstract vector containing the classes in the target variable

# Returns

  - `freq`: A float for the frequency of the reference class given the category
"""
function compute_label_freq_for_level(targets_for_level, y_classes)
    # Assumes binary classification where the first level is the reference class
    positive_class = y_classes[1]
    freq = sum(targets_for_level .== positive_class) / length(targets_for_level)
    return freq
end

"""
**Private method.**

Given the targets belonging to a specific category (level) for a categorical variable,
find the mean of these values (regression is assumed).

# Arguments

  - `targets_for_level`: An abstract vector containing the target Continuous or Count values

# Returns

  - `avg`: A float for the mean of the targets given the category
"""
function compute_target_mean_for_level(targets_for_level)
    avg = mean(targets_for_level)
    return avg
end

"""
**Private method.**

Given the targets belonging to a specific category (level) for a categorical variable,
find the frequency of each of the classes (multiclass classification assumed classification is assumed).

# Arguments

  - `targets_for_level`: An abstract vector containing the target Multiclass or OrderedFactor values
  - `y_classes`: An abstract vector containing the classes in the target variable

# Returns

  - `freqs`: A vector of floats for the frequency of the reference class
"""
function compute_label_freqs_for_level(targets_for_level, y_classes)
    # e.g., if y_classes = [1, 2, 3, 4]
    # then get the frequency of occurrence of each in targets_for_level
    freqs = [
        sum(targets_for_level .== y_level) / length(targets_for_level) for
        y_level in y_classes
    ]
    return freqs
end

"""
**Private method.**

Given the hyperparameter m, compute λ as in [Micci-Barreca, 2001] unless λ was overridden.
"""
function compute_shrinkage(targets_for_level; m = 0, λ = 1.0)
    # If λ has changed from the default then don't recompute
    if m != 0 && λ == 1.0
        ni = length(targets_for_level)
        λ = ni / (m + ni)
    end
    return λ
end

"""
**Private method.**

Compute m automatically using empirical Bayes estimation as suggested in [Micci-Barreca, 2001].
Only possible for regression tasks
"""
function compute_m_auto(task, targets_for_level; y_var)
    # call if m=:auto to compute by empirical Bayes est.
    task !== "Regression" && error("m = `auto` is only supported for regression")
    y_var_level = std(targets_for_level)^2
    m = y_var / y_var_level
    return m
end

"""
**Private method.**

Implement mixing between a posterior and a prior statistic  by computing `λ * posterior + (1 - λ) * prior`
"""
function mix_stats(; posterior, prior, λ)
    # mixing prior and posterior with mixing factor λ
    return λ .* posterior .+ (1 - λ) .* prior
    # prior is like the frequency of the reference class over the whole data
    # posterior is the frequency given rows that have the specific catefory
end


"""
**Private method.**

    target_encoder_fit(X, y, features=[]; ignore=true, ordered_factor=false, λ = 1.0, m=0)

Fit a target encoder on table X with target y by computing the necessary statistics for every categorical feature.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `y`:  An abstract vector of labels (e.g., strings) that correspond to the observations in X
  - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the features given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
  - `λ`: Shrinkage hyperparameter used to mix between posterior and prior statistics as described in [1]
  - `m`: An integer hyperparameter to compute shrinkage as described in [1]. If `m=:auto` then m will be computed using
    empirical Bayes estimation as described in [1]

# Returns

  - `cache`: A dictionary containing a dictionary `y_stat_given_feat_level` with the necessary statistics needed to transform
    every categorical feature as well as other metadata needed for transform.
"""
function target_encoder_fit(
    X,
    y::AbstractVector,
    features = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    lambda::Real = 1.0,
    m::Real = 0,
)
    # 1. Figure out task (classification or regression)
    y_scitype = elscitype(y)
    classification_types = (y_scitype <: Multiclass || y_scitype <: OrderedFactor)
    regression_types = (y_scitype <: Continuous || y_scitype <: Count)
    task =
        regression_types ? "Regression" :
        classification_types ? "Classification" : "Unknown"
    task == "Unknown" && error(
        "Your target must be Continuous/Count for regression or Multiclass/OrderedFactor for classification",
    )

    # 2. Setup prior statistics 
    if task == "Regression"
        y_mean = mean(y)                             # for mixing
        m == :auto && (y_var = std(y)^2)              # for empirical Bayes estimation
    else
        y_classes = levels(y)
        is_multiclass = length(y_classes) > 2
        if !is_multiclass       # binary case
            y_prior = sum(y .== y_classes[1]) / length(y)   # for mixing
        else                    # multiclass case
            y_stat_given_feat_level =
                y_priors = [sum(y .== y_level) / length(y) for y_level in y_classes]    # for mixing
        end
    end

    # 3. Define function to compute the new value(s) for each level given a column
    function feature_mapper(col, name)
        feat_levels = levels(col)
        y_stat_given_feat_level_for_col =
            Dict{eltype(feat_levels), Any}()
        for level in levels(col)
            # Get the targets of an example that belong to this level
            targets_for_level = y[col.==level]

            # Compute λ for mixing
            m == :auto && (m = compute_m_auto(task, targets_for_level; y_var = y_var))
            lambda = compute_shrinkage(targets_for_level; m = m, λ = lambda)

            if task == "Classification"
                if !is_multiclass           # 3.1 Binary classification
                    y_freq_for_level =
                        compute_label_freq_for_level(targets_for_level, y_classes)
                    y_stat_given_feat_level_for_col[level] =
                        mix_stats(
                            posterior = y_freq_for_level,
                            prior = y_prior,
                            λ = lambda,
                        )
                else                        # 3.2 Multiclass classification
                    y_freqs_for_level =
                        compute_label_freqs_for_level(targets_for_level, y_classes)
                    y_stat_given_feat_level_for_col[level] = mix_stats(
                        posterior = y_freqs_for_level,
                        prior = y_priors,
                        λ = lambda,
                    )
                end
            else                            # 3.3 Regression
                y_mean_for_level = compute_target_mean_for_level(targets_for_level)
                y_stat_given_feat_level_for_col[level] =
                    mix_stats(posterior = y_mean_for_level, prior = y_mean, λ = lambda)
            end
        end
        return y_stat_given_feat_level_for_col
    end

    # 4. Pass the function to generic_fit
    y_stat_given_feat_level, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,
    )

    cache = Dict(
        :task => task,
        :num_classes => (task == "Regression") ? -1 : length(y_classes),
        :y_stat_given_feat_level => y_stat_given_feat_level,
        :encoded_features => encoded_features,
        :y_classes => (task == "Regression") ? nothing : y_classes,
    )
    return cache
end



"""
**Private method.**

    target_encoder_transform(X, cache)

Transform given data with fitted target encoder cache.

# Arguments
- `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
`Multiclass` or `OrderedFactor`
- `cache`: A dictionary containing a dictionary `y_stat_given_feat_level` with the necessary statistics for 
every categorical feature as well as other metadata needed for transform

# Returns
- `X`: A table where the categorical features as specified during fitting are transformed by target encoding. Other features will remain
    the same. This will attempt to preserve the type of the table but may not succeed. 
"""

function target_encoder_transform(X, cache)
    task = cache[:task]
    y_stat_given_feat_level = cache[:y_stat_given_feat_level]
    num_classes = cache[:num_classes]
    y_classes = cache[:y_classes]

    return generic_transform(
        X,
        y_stat_given_feat_level;
        single_feat = task == "Regression" || (task == "Classification" && num_classes < 3),
        use_levelnames = true,
        custom_levels = y_classes,)
end

