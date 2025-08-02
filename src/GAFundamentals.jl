module GAFundamentals
import Comonicon
import Comonicon.Arg: Path
using TimesDates
using Dates
using Parquet
using DataFrames
using Metaheuristics
import JSON
using Tables
include("utils.jl")
using .Utils
import Plots: savefig


const THRESHOLD = 0.5

const Enabled = String


function get_pandas_attr(file::Parquet.File)::Dict{String, Dict{String, String}}
    meta = file.meta.key_value_metadata
    for kv in meta
        if kv.key == "PANDAS_ATTRS"
            return JSON.parse(String(kv.value))
        end
    end
    return Dict{String, Dict{String, String}}()
end


function read(input::Parquet.File)::DataFrame
    table = read_parquet(input.path)
    df = DataFrame(table)
    return df
end


"""
    find_at(df::AbstractDataFrame, points::Vector{Any}, a::Symbol, b::Symbol)

Extract values from column `a` where column `b` contains values matching elements in `points`

# Arguments
- `df::AbstractDataFrame`: The DataFrame to extract data from
- `points::Vector{Any}`: Values to filter column `b` by
- `a::Symbol`: Column to extract values from
- `b::Symbol`: Column to filter and sort by

# Returns
- Vector of values from column `a` corresponding to rows where column `b` matches 
  any value in `points`

# Warning
- The function will fail if df is not sorted by `b`

# Examples

```julia
df = DataFrame(id = 1:5, value = ["A", "B", "C", "B", "A"])
find_at(df, ["A", "B"], :id, :value) # Returns [1, 5, 2, 4] (first "A" rows, then "B" rows)
```
"""
function find_at(
        df::AbstractDataFrame,
        points::AbstractVector,
        a::Symbol,
        b::Symbol
    )::AbstractVector
    @assert issorted(df, b)
    result = []

    for point in points
        # Find all rows matching this point
        matching_rows = filter(row -> row[b] == point, df)
        if !isempty(matching_rows)
            # Add the values from column a
            append!(result, matching_rows[:, a])
        end
    end

    return result
end

function fitness(
        prices::DataFrame,
        fundamentals::DataFrame,
        groups::Dict{String, String};
        along::Bool = false,
    )
    gx = [0.0]
    hx = [0.0]
    allnames = names(fundamentals)
    fundamental_names = get_fundamental_names(allnames)
    years = unique(sort(year.(prices[:, :Date])))[2:end]

    n_years = length(years)

    return function (space)
        fundamentals_mask = space .> THRESHOLD
        colnames = fundamental_names[fundamentals_mask]

        filtered_groups = Dict(k => v for (k, v) in groups if k in colnames)


        if isempty(filtered_groups)
            if along
                return (fill(1000.0, n_years * 2), gx, hx)
            end
            return ([1000.0, 1000.0], gx, hx)
        end
        rank = compute_rank(
            fundamentals[:, [["Ticker", "Year"]..., colnames...]],
            filtered_groups
        )
        tn = top_n(rank, 3)
        bn = bottom_n(rank, 3)

        top_mirrored = mirror(prices, tn)
        bottom_mirrored = mirror(prices, bn)
        transform!(top_mirrored, :Date => ByRow(date -> year(date)) => :Year)
        transform!(bottom_mirrored, :Date => ByRow(date -> year(date)) => :Year)
        rt = combine(groupby(top_mirrored, :Year), returns)
        rb = combine(groupby(bottom_mirrored, :Year), returns)
        pt = combine(groupby(rt, :Year), portfolio_returns)
        pb = combine(groupby(rb, :Year), portfolio_returns)

        ct = cum_returns_strategies(groupby(pt, :Year))
        cb = cum_returns_strategies(groupby(pb, :Year))

        if !along
            fx1 = last(ct[:, :Close])
            fx2 = last(cb[:, :Close])
            return [fx1, fx2], gx, hx
        end
        points_t = Vector{TimeDate}(undef, n_years)
        points_b = Vector{TimeDate}(undef, n_years)
        for (i, y) in enumerate(years)
            points_t[i] = maximum(filter(row -> year(row.Date) == y, ct).Date)
            points_b[i] = maximum(filter(row -> year(row.Date) == y, cb).Date)
        end
        @assert length(points_t) == length(points_b)
        fxs1 = find_at(
            ct,
            points_t,
            :Close,
            :Date
        )
        fxs2 = find_at(
            cb,
            points_b,
            :Close,
            :Date
        )
        @assert length(fxs1) == length(fxs2)

        fx = Vector{Float64}(undef, n_years * 2)
        fx[1:length(fxs1)] = .-fxs1
        fx[(length(fxs1) + 1):end] = fxs2

        return fx, gx, hx
    end
end

function save_metadata(original_columns::AbstractVector{String}, optimal_columns::AbstractVector{String}, path::String)
    d = Dict("optimal" => optimal_columns, "original" => original_columns)
    return open(path, "w") do f
        JSON.print(f, d)
    end
end

"""
# Arguments
- `prices::DataFrame`: The DataFrame with prices
- `fundamentals::DataFrame`: The fundamentals DataFrame
- `fundamental_groups::Dict{String, String}`: A mapping of each fundamentals column to its associated group 
- `output::AbstractString`: The place where to save
- `verbose::Bool`: Whether to display the progress in the optimization 
- `user_solutions::Union{DataFrame, Nothing}`: A Bit DataFrame, 1 demonstrate presence, 0 absence of feature

"""
function run(
        prices::DataFrame,
        fundamentals::DataFrame,
        fundamental_groups::Dict{
            String,
            String,
        },
        output::AbstractString,
        verbose::Bool,
        user_solutions::Union{DataFrame, Nothing},
        f_calls_limit,
        iterations::Int,
        time_limit::Float64,
        along::Bool,
    )
    # NOTE: Remove 2 to ignore Year and Ticker column
    n_features = ncol(fundamentals) - 2
    lb = zeros(n_features)
    ub = ones(n_features)
    bounds = BoxConstrainedSpace(lb = lb, ub = ub)

    algo = NSGA2(
        # N = 200,
        # p_cr = 0.5,
        # p_m = 0.5,
        # n_cr = 20,
        # n_m = 10,
        options = Options(
            verbose = verbose,
            f_calls_limit = f_calls_limit,
            iterations = iterations,
            time_limit = time_limit
        )
    )
    f = fitness(prices, fundamentals, fundamental_groups; along = along)
    if !isnothing(user_solutions)
        @assert ncol(user_solutions) == n_features
        user_solutions = replace(v -> Bool(v) ? THRESHOLD : 0, Matrix(float.(user_solutions)))

        set_user_solutions!(algo, user_solutions, f)
    end


    result = optimize(
        f,
        bounds,
        algo
    )
    optimal_mask = minimizer(result) .> THRESHOLD
    fundamental_names = get_fundamental_names(names(fundamentals))
    colnames = fundamental_names[optimal_mask]

    p1 = plot_portfolio(prices, fundamentals, fundamental_groups)
    save_metadata(
        fundamental_names,
        colnames,
        joinpath(output, "data.json")
    )
    savefig(p1, joinpath(output, "notoptimized.png"))
    filtered_groups = Dict(k => v for (k, v) in fundamental_groups if k in colnames)
    selected_columns = [["Year", "Ticker"]..., colnames...]
    p2 = plot_portfolio(
        prices,
        fundamentals[:, selected_columns],
        filtered_groups
    )
    return savefig(p2, joinpath(output, "optimized.png"))
end


@Comonicon.main function main(
        prices_path::Path,
        fundamentals_path::Path,
        output::Path;
        user_sol_path::Path = Path(""),
        verbose::Bool = false,
        f_calls_limit = 10000.0,
        iterations::Int = 1000,
        time_limit::Float64 = 60.0,
        along::Bool = false
    )
    prices_input = Parquet.File(prices_path.content)
    prices = read(prices_input)
    prices[!, :Date] = epochns_to_datetime.(prices[:, :Date])
    fundamentals_input = Parquet.File(fundamentals_path.content)
    fundamentals = read(fundamentals_input)
    fundamentals[!, :Year] = epochns_to_datetime.(fundamentals[:, :Year])
    fundamentals_groups = get_pandas_attr(fundamentals_input)["columns"]


    user_solutions = user_sol_path == Path("") ? nothing : DataFrame(
            read_parquet(user_sol_path.content)
        )

    run(
        prices,
        fundamentals,
        fundamentals_groups,
        output.content,
        verbose,
        user_solutions,
        f_calls_limit,
        iterations,
        time_limit,
        along
    )
end
end
