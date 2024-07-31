
mutable struct UnivariateTimeTypeToContinuous <: Unsupervised
    zero_time::Union{Nothing, TimeType}
    step::Period
end

function UnivariateTimeTypeToContinuous(;
    zero_time=nothing, step=Dates.Hour(24))
    model = UnivariateTimeTypeToContinuous(zero_time, step)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::UnivariateTimeTypeToContinuous)
    # Step must be able to be added to zero_time if provided.
    msg = ""
    if model.zero_time !== nothing
        try
            tmp = model.zero_time + model.step
        catch err
            if err isa MethodError
                model.zero_time, model.step, status, msg = _fix_zero_time_step(
                    model.zero_time, model.step)
                if status === :error
                    # Unable to resolve, rethrow original error.
                    throw(err)
                end
            else
                throw(err)
            end
        end
    end
    return msg
end

function _fix_zero_time_step(zero_time, step)
    # Cannot add time parts to dates nor date parts to times.
    # If a mismatch is encountered. Conversion from date parts to time parts
    # is possible, but not from time parts to date parts because we cannot
    # represent fractional date parts.
    msg = ""
    if zero_time isa Dates.Date && step isa Dates.TimePeriod
        # Convert zero_time to a DateTime to resolve conflict.
        if step % Hour(24) === Hour(0)
            # We can convert step to Day safely
            msg = "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `step` to `Day`."
            step = convert(Day, step)
        else
            # We need datetime to be compatible with the step.
            msg = "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `zero_time` to `DateTime`."
            zero_time = convert(DateTime, zero_time)
        end
        return zero_time, step, :success, msg
    elseif zero_time isa Dates.Time && step isa Dates.DatePeriod
        # Convert step to Hour if possible. This will fail for
        # isa(step, Month)
        msg = "Cannot add `DatePeriod` `step` to `Time` `zero_time`. Converting `step` to `Hour`."
        step = convert(Hour, step)
        return zero_time, step, :success, msg
    else
        return zero_time, step, :error, msg
    end
end

function MMI.fit(model::UnivariateTimeTypeToContinuous, verbosity::Int, X)
    if model.zero_time !== nothing
        min_dt = model.zero_time
        step = model.step
        # Check zero_time is compatible with X
        example = first(X)
        try
            X - min_dt
        catch err
            if err isa MethodError
                @warn "`$(typeof(min_dt))` `zero_time` is not compatible with `$(eltype(X))` vector. Attempting to convert `zero_time`."
                min_dt = convert(eltype(X), min_dt)
            else
                throw(err)
            end
        end
    else
        min_dt = minimum(X)
        step = model.step
        message = ""
        try
            min_dt + step
        catch err
            if err isa MethodError
                min_dt, step, status, message = _fix_zero_time_step(min_dt, step)
                if status === :error
                    # Unable to resolve, rethrow original error.
                    throw(err)
                end
            else
                throw(err)
            end
        end
        isempty(message) || @warn message
    end
    cache = nothing
    report = NamedTuple()
    fitresult = (min_dt, step)
    return fitresult, cache, report
end

function MMI.transform(model::UnivariateTimeTypeToContinuous, fitresult, X)
    min_dt, step = fitresult
    if typeof(min_dt) ≠ eltype(X)
        # Cannot run if eltype in transform differs from zero_time from fit.
        throw(ArgumentError("Different `TimeType` encountered during `transform` than expected from `fit`. Found `$(eltype(X))`, expected `$(typeof(min_dt))`"))
    end
    # Set the size of a single step.
    next_time = min_dt + step
    if next_time == min_dt
        # Time type loops if step is a multiple of Hour(24), so calculate the
        # number of multiples, then re-scale to Hour(12) and adjust delta to match original.
        m = step / Dates.Hour(12)
        delta = m * (
            Float64(Dates.value(min_dt + Dates.Hour(12)) - Dates.value(min_dt)))
    else
        delta = Float64(Dates.value(min_dt + step) - Dates.value(min_dt))
    end
    return @. Float64(Dates.value(X - min_dt)) / delta
end

metadata_model(UnivariateTimeTypeToContinuous,
    input_scitype   = AbstractVector{<:ScientificTimeType},
    output_scitype = AbstractVector{Continuous},
    human_name ="single variable transformer that creates "*
         "continuous representations of temporally typed data",
    load_path    = "MLJModels.UnivariateTimeTypeToContinuous")

"""
$(MLJModelInterface.doc_header(UnivariateTimeTypeToContinuous))

Use this model to convert vectors with a `TimeType` element type to
vectors of `Float64` type (`Continuous` element scitype).


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector whose element type is a subtype of
  `Dates.TimeType`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `zero_time`: the time that is to correspond to 0.0 under
  transformations, with the type coinciding with the training data
  element type. If unspecified, the earliest time encountered in
  training is used.

- `step::Period=Hour(24)`: time interval to correspond to one unit
  under transformation


# Operations

- `transform(mach, xnew)`: apply the encoding inferred when `mach` was fit


# Fitted parameters

`fitted_params(mach).fitresult` is the tuple `(zero_time, step)`
actually used in transformations, which may differ from the
user-specified hyper-parameters.


# Example

```
using MLJ
using Dates

x = [Date(2001, 1, 1) + Day(i) for i in 0:4]

encoder = UnivariateTimeTypeToContinuous(zero_time=Date(2000, 1, 1),
                                         step=Week(1))

mach = machine(encoder, x)
fit!(mach)
julia> transform(mach, x)
5-element Vector{Float64}:
 52.285714285714285
 52.42857142857143
 52.57142857142857
 52.714285714285715
 52.857142
```

"""
UnivariateTimeTypeToContinuous