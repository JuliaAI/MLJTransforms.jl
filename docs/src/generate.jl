function generate(dir; execute = true, pluto = false)
    quote
        using Pkg
        Pkg.activate(temp = true)
        Pkg.add("Literate")
        using Literate

        OUTDIR = $dir
        outdir = splitpath(OUTDIR)[end]
        INFILE = joinpath(OUTDIR, "notebook.jl")

        @info "Generating notebooks for $outdir. "

        Literate.markdown(
            INFILE,
            OUTDIR,
            execute = true,
            # Use regular julia code blocks instead of @example to prevent execution by Documenter
            config = Dict("codefence" => Pair("````julia", "````")),
        )


    end |> eval
end
