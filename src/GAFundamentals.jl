module GAFundamentals
import Comonicon
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


THRESHOLD = 0.5


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


function fitness(prices::DataFrame, fundamentals::DataFrame, groups::Dict{String, String})
    return function (space)
        fundamentals_mask = space .> THRESHOLD
        allnames = names(fundamentals)
        fundamental_names = get_fundamental_names(allnames)
        colnames = fundamental_names[fundamentals_mask]

        filtered_groups = Dict(k => v for (k, v) in groups if k in colnames)

        if isempty(filtered_groups)
            # NOTE: Necessary to return early because sum_scores can not take
            # an empty dictionary
            # Penalize to minimize those solutions(only 1 anyway)
            return [1000.0, 1000.0], [0.0], [0.0]  # Large penalty values
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

        fx1 = last(ct[:, :Close])
        fx2 = last(cb[:, :Close])
        fx = [-fx1, fx2] # fx1 is negated to turn it into maximization
        gx = [0.0]
        hx = [0.0]
        return fx, gx, hx
    end
end

function save_metadata(original_columns::AbstractVector{String}, optimal_columns::AbstractVector{String}, path::String)
    d = Dict("optimal" => optimal_columns, "original" => original_columns)
    return open(path, "w") do f
        JSON.print(f, d)
    end
end

function run(
        prices::DataFrame,
        fundamentals::DataFrame,
        fundamental_groups::Dict{
            String,
            String,
        },
        output::AbstractString,
        verbose::Bool,
        f_calls_limit = 10000.0,
        iterations::Int = 1000,
        time_limit::Float64 = 60.0
    )
    n_features = ncol(fundamentals) - 2 # Remove 2 to ignore Year and Ticker column
    lb = zeros(n_features)
    ub = ones(n_features)
    bounds = BoxConstrainedSpace(lb = lb, ub = ub)
    result = optimize(
        fitness(prices, fundamentals, fundamental_groups),
        bounds,
        NSGA2(
            options = Options(
                verbose = verbose,
                f_calls_limit = f_calls_limit,
                iterations = iterations,
                time_limit = time_limit
            )
        )
    )
    optimal_mask = minimizer(result) .> THRESHOLD
    fundamental_names = get_fundamental_names(names(fundamentals))
    colnames = fundamental_names[optimal_mask]

    p1 = plot_portfolio(prices, fundamentals, fundamental_groups)
    save_metadata(fundamental_names, colnames, joinpath(output, "data.json"))
    savefig(p1, joinpath(output, "notoptimized.png"))
    filtered_groups = Dict(k => v for (k, v) in fundamental_groups if k in colnames)
    selected_columns = [["Year", "Ticker"]..., colnames...]
    p2 = plot_portfolio(prices, fundamentals[:, selected_columns], filtered_groups)
    return savefig(p2, joinpath(output, "optimized.png"))
end


@Comonicon.main function main(
        prices_path::AbstractString,
        fundamentals_path::AbstractString,
        output::AbstractString;
        verbose::Bool = false,
        f_calls_limit = 10000.0,
        iterations::Int = 1000,
        time_limit::Float64 = 60.0
    )
    prices_input = Parquet.File(prices_path)
    prices = read(prices_input)
    prices[!, :Date] = epochns_to_datetime.(prices[:, :Date])
    fundamentals_input = Parquet.File(fundamentals_path)
    fundamentals = read(fundamentals_input)
    fundamentals[!, :Year] = epochns_to_datetime.(fundamentals[:, :Year])
    fundamentals_groups = get_pandas_attr(fundamentals_input)["columns"]

    run(prices, fundamentals, fundamentals_groups, output, verbose, f_calls_limit, iterations, time_limit)
end
end
