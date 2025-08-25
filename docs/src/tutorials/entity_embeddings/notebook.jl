# # Entity Embeddings Tutorial

# **Julia version** is assumed to be 1.10.*

# This demonstration is available as a Jupyter notebook or julia script (as well as the dataset)
# [here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/common_workflows/entity_embeddings).
#
# Entity embedding is a newer deep learning approach for categorical encoding introduced in 2016 by Cheng Guo and Felix Berkhahn. 
# It employs a set of embedding layers to map each categorical feature into a dense continuous vector in a similar fashion to how they are employed in NLP architectures.
#
# In MLJFlux, the `EntityEmbedder` provides a high-level interface to learn entity embeddings using any supervised MLJFlux model as the underlying learner.
# The embedder can be used as a transformer in MLJ pipelines to encode categorical features with learned embeddings, which can then be used as features in downstream machine learning models.
#
# In this tutorial, we will explore how to use the `EntityEmbedder` to learn and apply entity embeddings on the Google Play Store dataset.
#
# ## Learning Objectives
# - Understand the concept of entity embeddings for categorical encoding
# - Learn how to use `EntityEmbedder` from MLJFlux
# - Apply entity embeddings to a real-world dataset
# - Visualize the learned embedding spaces
# - Build pipelines combining embeddings with downstream models

using Pkg;
Pkg.activate(@__DIR__);
Pkg.instantiate(); #src



## Import all required packages
using MLJ
using CategoricalArrays
using DataFrames
using Optimisers
using Random
using Tables
using ProgressMeter
using Plots
using ScientificTypes
using CSV
using StatsBase  ## For countmap
import Plots: mm  ## For margin units

# ## Data Loading and Preprocessing
#
# We'll use the Google Play Store dataset which contains information about mobile applications.
# This dataset has several categorical features that are perfect for demonstrating entity embeddings:
# - **Category**: App category (e.g., Games, Social, Tools)
# - **Content Rating**: Age rating (e.g., Everyone, Teen, Mature)
# - **Genres**: Primary genre of the app
# - **Android Ver**: Required Android version
# - **Type**: Free or Paid

## Load the Google Play Store dataset
df = CSV.read("./googleplaystore.csv", DataFrame);

# ### Data Cleaning and Type Conversion
#
# The raw dataset requires significant cleaning. We'll handle:
# 1. **Reviews**: Convert to integers
# 2. **Size**: Parse size strings like "14M", "512k" to numeric values
# 3. **Installs**: Remove formatting characters and convert to integers  
# 4. **Price**: Remove dollar signs and convert to numeric
# 5. **Genres**: Extract primary genre only

## Custom parsing function that returns missing instead of nothing
safe_parse(T, s) = something(tryparse(T, s), missing);

## Reviews: ensure integer
df.Reviews = safe_parse.(Int, string.(df.Reviews));

## Size: "14M", "512k", or "Varies with device"
function parse_size(s)
    if s == "Varies with device"
        return missing
    elseif occursin('M', s)
        return safe_parse(Float64, replace(s, "M" => "")) * 1_000_000
    elseif occursin('k', s)
        return safe_parse(Float64, replace(s, "k" => "")) * 1_000
    else
        return safe_parse(Float64, s)
    end
end
df.Size = parse_size.(string.(df.Size));

## Installs: strip '+' and ',' then parse
clean_installs = replace.(string.(df.Installs), r"[+,]" => "")
df.Installs = safe_parse.(Int, clean_installs);

## Price: strip leading '$'
df.Price = safe_parse.(Float64, replace.(string.(df.Price), r"^\$" => ""));

## Genres: take only the primary genre
df.Genres = first.(split.(string.(df.Genres), ';'));

# ### Storing Category Information for Visualization
#
# We'll store the unique values of each categorical feature to use later when visualizing the embeddings.

## Store unique category names for visualization later
category_names = Dict(
    :Category => sort(unique(df.Category)),
    Symbol("Content Rating") => sort(unique(df[!, Symbol("Content Rating")])),
    :Genres => sort(unique(df.Genres)),
    Symbol("Android Ver") => sort(unique(df[!, Symbol("Android Ver")])),
);

println("Category names extracted:")
for (feature, names) in category_names
    println("$feature: $(length(names)) categories")
end
# ### Feature Selection and Missing Value Handling
#
# We'll select the most relevant features and remove any rows with missing values to ensure clean data for our embedding model.

select!(
    df,
    [
        :Category, :Reviews, :Size, :Installs, :Type,
        :Price, Symbol("Content Rating"), :Genres, Symbol("Android Ver"), :Rating,
    ],
);
dropmissing!(df);

# ### Creating Categorical Target Variable
#
# For this tutorial, we'll convert the continuous rating into a categorical classification problem.
# This will allow us to use a classification model that can learn meaningful embeddings.
#
# We'll create 10 rating categories by rounding to the nearest 0.5 (e.g., 0.0, 0.5, 1.0, ..., 4.5, 5.0).

## Create 10 classes: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
function rating_to_categorical(rating)
    ## Clamp rating to valid range and round to nearest 0.5
    clamped_rating = clamp(rating, 0.0, 5.0)
    rounded_rating = round(clamped_rating * 2) / 2  ## Round to nearest 0.5
    return string(rounded_rating)
end

## Apply the transformation
df.RatingCategory = categorical([rating_to_categorical(r) for r in df.Rating]);

## Check the distribution of categorical rating labels
println("Distribution of categorical rating labels:")
println(sort(countmap(df.RatingCategory)))
println("\nUnique rating categories: $(sort(unique(df.RatingCategory)))")

# ### Type Coercion for MLJ
#
# MLJ requires explicit type coercion to understand which columns are categorical vs continuous.
# This step is crucial for the `EntityEmbedder` to identify which features need embedding layers.

## Coerce types for MLJ compatibility
df = coerce(df,
    :Category => Multiclass,
    :Reviews => Continuous,
    :Size => Continuous,
    :Installs => Continuous,
    :Type => Multiclass,
    :Price => Continuous,
    Symbol("Content Rating") => Multiclass,
    :Genres => Multiclass,
    Symbol("Android Ver") => Multiclass,
    :Rating => Continuous,  ## Keep original for reference
    :RatingCategory => Multiclass,  ## New categorical target
);
schema(df)

# ### Data Splitting
#
# We'll split our data into training and testing sets using stratified sampling to ensure balanced representation of rating categories.

## Split into features and target
y = df[!, :RatingCategory]  ## Use categorical rating as target
X = select(df, Not([:Rating, :RatingCategory]));  ## Exclude both rating columns from features

## Split the data with stratification
(X_train, X_test), (y_train, y_test) = partition(
    (X, y),
    0.8,
    multi = true,
    shuffle = true,
    stratify = y,
    rng = Random.Xoshiro(41),
);

using MLJFlux

# ## Building the EntityEmbedder Model

## Load the neural network classifier
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg = MLJFlux

# ### Configuring the Base Neural Network
#
# We'll create a neural network classifier with custom embedding dimensions for each categorical feature.
# Setting smaller embedding dimensions (like 2D) makes it easier to visualize the learned representations.

## Create the underlying supervised model that will learn the embeddings
base_clf = NeuralNetworkClassifier(
    builder = MLJFlux.Short(n_hidden = 14),
    optimiser = Optimisers.Adam(10e-2),
    batch_size = 20,
    epochs = 5,
    acceleration = CUDALibs(),
    embedding_dims = Dict(
        :Category => 2,
        :Type => 2,
        Symbol("Content Rating") => 2,
        :Genres => 2,
        Symbol("Android Ver") => 2,
    ),
    rng = 39,
);

# ### Creating the EntityEmbedder
#
# The `EntityEmbedder` wraps our neural network and can be used as a transformer in MLJ pipelines.
# By default, it uses `min(n_categories - 1, 10)` dimensions for any categorical feature not explicitly specified.

## Create the EntityEmbedder using the neural network
embedder = EntityEmbedder(base_clf)

# ## Training the EntityEmbedder
#
# Now we'll train the embedder on our training data. The model learns to predict app ratings while simultaneously learning meaningful embeddings for categorical features.

## Create and train the machine
mach = machine(embedder, X_train, y_train)
MLJ.fit!(mach, force = true, verbosity = 1);

# ### Transforming Data with Learned Embeddings
#
# After training, we can use the embedder as a transformer to convert categorical features into their learned embedding representations.

## Transform the data using the learned embeddings
X_train_embedded = MLJFlux.transform(mach, X_train)
X_test_embedded = MLJFlux.transform(mach, X_test);

## Check the schema transformation
println("Original schema:")
schema(X_train)
println("\nEmbedded schema:")
schema(X_train_embedded)
X_train_embedded; #hide

# ## Using Embeddings in ML Pipelines
#
# One of the key advantages of entity embeddings is that they can be used as features in any downstream machine learning model.
# Let's create a pipeline that combines our `EntityEmbedder` with a k-nearest neighbors classifier.

## Load KNN classifier
KNNClassifier = @load KNNClassifier pkg = NearestNeighborModels

## Create a pipeline: EntityEmbedder -> KNNClassifier
pipe = embedder |> KNNClassifier(K = 5);

## Train the pipeline
pipe_mach = machine(pipe, X_train, y_train)
MLJ.fit!(pipe_mach, verbosity = 0)

# ## Visualizing the Learned Embedding Spaces
#
# One of the most powerful aspects of entity embeddings is their interpretability. Since we used 2D embeddings, we can visualize how the model has organized different categories in the embedding space.

## Extract the learned embedding matrices from the fitted model
mapping_matrices = fitted_params(mach)[4]

# ### Creating Embedding Visualization Function
#
# We'll create a helper function to plot the 2D embedding space for each categorical feature.
# Each point represents a category, and its position shows how the model learned to represent it.

## Function to create and display scatter plot for categorical embeddings
function plot_categorical_embeddings(feature_name, feature_categories, embedding_matrix)
    ## Convert feature_name to string to handle both Symbol and String inputs
    feature_name_str = string(feature_name)

    ## Create scatter plot for this feature's embeddings
    p = scatter(embedding_matrix[1, :], embedding_matrix[2, :],
        title = "$(feature_name_str) Embeddings",
        xlabel = "Dimension 1",
        ylabel = "Dimension 2",
        label = "$(feature_name_str)",
        legend = :topright,
        markersize = 8,
        size = (1200, 600))

    ## Annotate each point with the actual category name
    for (i, col) in enumerate(eachcol(embedding_matrix))
        if i <= length(feature_categories)
            cat_name = string(feature_categories[i])
            ## Truncate long category names for readability
            display_name = length(cat_name) > 10 ? cat_name[1:10] * "..." : cat_name
            annotate!(p, col[1] + 0.02, col[2] + 0.02, text(display_name, :black, 5))
        end
    end

    ## Save the plot
    savefig(p, "embedding_$(lowercase(replace(feature_name_str, " " => "_"))).png") #hide

    ## Display the plot
    display(p) #hide
    println("Displayed embedding plot for: $(feature_name_str)") #hide
    return p
end;

# ### Generating Embedding Plots for Each Categorical Feature
#
# Let's visualize the embedding space for each of our categorical features to understand what patterns the model learned.

## Create separate plots for each categorical feature's embeddings

## Plot 1: Category embeddings
plot_categorical_embeddings(
    :Category,
    category_names[:Category],
    mapping_matrices[:Category],
);
#md # ![Category Embeddings](embedding_category.png)
# Notice that pairs such as social and entertainment, shopping and finance, and comics and art are closer together than others.

## Plot 2: Content Rating embeddings
plot_categorical_embeddings(
    Symbol("Content Rating"),
    category_names[Symbol("Content Rating")],
    mapping_matrices[Symbol("Content Rating")],
);
#md # ![Content Rating Embeddings](embedding_content_rating.png)
# The `Everyone` category is positioned far from all others.

## Plot 3: Genres embeddings
plot_categorical_embeddings(:Genres, category_names[:Genres], mapping_matrices[:Genres]);
#md # ![Genres Embeddings](embedding_genres.png)
# Here the results may be less interpretable; the idea is that for purposes of indetifying the rating, the model considered categories closer together as more similar.

## Plot 4: Android Ver embeddings
plot_categorical_embeddings(
    Symbol("Android Ver"),
    category_names[Symbol("Android Ver")],
    mapping_matrices[Symbol("Android Ver")],
);
#md # ![Android Ver Embeddings](embedding_android_ver.png)
# Clear patterns like close proximity between (7.1 and up) and, 7.0-7.1

## Plot 5: Type embeddings (if it exists in the mapping)
plot_categorical_embeddings(:Type, sort(unique(df.Type)), mapping_matrices[:Type]);
#md # ![Type Embeddings](embedding_type.png)
# Indeed, `Free` and `Paid` are too dissimilar.

# This demonstrates the power of entity embeddings as a modern approach to categorical feature encoding that goes beyond traditional methods like one-hot encoding or label encoding.
