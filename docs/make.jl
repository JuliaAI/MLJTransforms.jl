using Documenter
using MLJTransforms

DocMeta.setdocmeta!(MLJTransforms, :DocTestSetup, :(using MLJTransforms); recursive = true)

makedocs(
    sitename = "MLJTransforms",
    format = Documenter.HTML(
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(
                "https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap",
                class = :css,
            ),
            asset(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
                class = :css,
            ),
        ],
        repolink = "https://github.com/JuliaAI/MLJTransforms.jl",
    ),
    modules = [MLJTransforms],
    warnonly = true,
    pages = [
        "Introduction" => "index.md",
        "Transformers" => Any[
            "Numerical Transformers"=>"transformers/numerical.md",
            "Classical Encoders"=>"transformers/classical.md",
            "Neural-based Encoders"=>"transformers/neural.md",
            "Contrast Encoders"=>"transformers/contrast.md",
            "Utility Encoders"=>"transformers/utility.md",
            "Other Transformers"=>"transformers/others.md",
            "API Index" => "transformers/all_transformers.md",
        ],
        "Extended Examples" => Any[
            "Tutorial A" => "tutorials/T1.md",
            "Tutorial B" => "tutorials/T1.md",
        ],
        "Contributing" => "contributing.md"],
    doctest = false,
)

# Documenter can also automatically deploy documentation to gh-pages.
deploydocs(repo = "github.com/JuliaAI/MLJTransforms.jl.git", devbranch = "dev")
