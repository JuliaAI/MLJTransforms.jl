using Documenter
using MLJTransforms
using MLJFlux

DocMeta.setdocmeta!(MLJTransforms, :DocTestSetup, :(using MLJTransforms); recursive = true)
DocMeta.setdocmeta!(MLJFlux, :DocTestSetup, :(using MLJFlux); recursive = true)
makedocs(
    sitename = "MLJTransforms",
    format = Documenter.HTML(
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(
                "https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap",
                class = :css,
            ),
            asset(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
                class = :css,
            ),
        ],
        repolink = "https://github.com/JuliaAI/MLJTransforms.jl",
    ),
    modules = [MLJTransforms, MLJFlux],
    warnonly = true,
    pages = [
        "Introduction" => "index.md",
        "Transformers" => Any[
            "All Transformers"=>"transformers/all_transformers.md",
            "Encoders"=>Any[
                "Classical Encoders"    => "transformers/classical.md",
                "Neural-based Encoders" => "transformers/neural.md",
                "Contrast Encoders"     => "transformers/contrast.md",
                "Utility Encoders"      => "transformers/utility.md",
            ],
        ],
        "Extended Examples" => Any[
            "Standardization Impact"=>"tutorials/standardization/notebook.md",
            "Milk Quality Classification"=>"tutorials/classic_comparison/notebook.md",
            "Wine Quality Prediction"=>"tutorials/wine_example/notebook.md",
            "Entity Embeddings Tutorial"=>"tutorials/entity_embeddings/notebook.md",
        ],
        "Contributing" => "contributing.md",
        "About" => "about.md",
    ],
    doctest = false,
)

deploydocs(repo = "github.com/JuliaAI/MLJTransforms.jl.git", devbranch = "dev")
