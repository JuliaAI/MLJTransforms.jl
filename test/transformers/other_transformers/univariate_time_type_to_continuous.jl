
@testset "TimeTypeToContinuous" begin
    let dt = [Date(2018, 6, 15) + Day(i) for i=0:10],
        transformer = UnivariateTimeTypeToContinuous(; step=Day(1))
        fr, _, _ = MLJBase.fit(transformer, 1, dt)
        @test fr == (Date(2018, 6, 15), Day(1))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10))
    end

    let dt = [Date(2018, 6, 15) + Day(i) for i=0:10],
        transformer = UnivariateTimeTypeToContinuous()
        fr, _, _ = @test_logs(
            (:warn, r"Cannot add `TimePeriod` `step`"),
            MLJBase.fit(transformer, 1, dt)
        )
        fr, _, _ = @test_logs (:warn, r"C") MLJBase.fit(transformer, 1, dt)
        @test fr == (Date(2018, 6, 15), Day(1))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10))
    end

    let dt = [Time(0, 0, 0) + Hour(i) for i=0:3:30],
        transformer = UnivariateTimeTypeToContinuous(;
            step = Hour(1),
            zero_time = Time(7, 0, 0),
        )
        fr, _, _ = MLJBase.fit(transformer, 1, dt)
        @test fr == (Time(7, 0, 0), Hour(1))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        ex = collect(0:3:30) .% 24 .- 7.0
        diff = map(dt_continuous .- ex) do d
            mod(d, 24.0)
        end
        @test all(diff .≈ 0.0)
    end

    let dt = [Time(0, 0, 0) + Hour(i) for i=0:3:30],
        transformer = UnivariateTimeTypeToContinuous()
        fr, _, _ = MLJBase.fit(transformer, 1, dt)
        @test fr == (Time(0, 0, 0), Hour(24))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        ex = collect(0:3:30) .% 24 ./ 24
        diff = map(dt_continuous .- ex) do d
            mod(d, 1.0)
        end
        @test all(diff .≈ 0.0)
    end

    # test log messages
    let dt = [DateTime(2018, 6, 15) + Day(i) for i=0:10],
        step=Hour(1),
        zero_time=Date(2018, 6, 15),
        transformer = @test_logs(
            (:warn, "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `zero_time` to `DateTime`."),
            UnivariateTimeTypeToContinuous(;
                step=step,
                zero_time=zero_time,
            )
        )
        fr, _, _ = MLJBase.fit(transformer, 1, dt)

        @test fr == (zero_time, step)
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10).*24)
    end

    let dt = [Time(0, 0, 0) + Hour(i) for i=0:3:30],
        zero_time=Time(0, 0, 0),
        step=Day(1),
        transformer = @test_logs(
            (:warn, "Cannot add `DatePeriod` `step` to `Time` `zero_time`. Converting `step` to `Hour`."),
            UnivariateTimeTypeToContinuous(;
                step=step,
                zero_time=zero_time,
            )
        )
        fr, _, _ = MLJBase.fit(transformer, 1, dt)

        @test fr == (zero_time, convert(Hour, step))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        ex = Float64.((0:3:30) .% 24)./24
        diff = map(dt_continuous .- ex) do d
            mod(d, 1.0)
        end
        @test all(diff .≈ 0.0)
    end

    let dt = [DateTime(2018, 6, 15) + Day(i) for i=0:10],
        step=Day(1),
        zero_time=Date(2018, 6, 15),
        transformer = UnivariateTimeTypeToContinuous(;
            step=step,
            zero_time=zero_time,
        )
        fr, _, _ = @test_logs(
            (:warn, r"`Date"),
            MLJBase.fit(transformer, 1, dt)
        )

        @test fr == (zero_time, step)
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10))
    end
end
