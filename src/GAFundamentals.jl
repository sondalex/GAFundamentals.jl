module GAFundamentals
import Comonicon
using TimesDates
using Dates
using Parquet
using DataFrames
using Metaheuristics
import JSON
using Tables
using Statistics
using StatsBase
using Plots

THRESHOLD = 0.5

function mirror(prices::DataFrame, group_n::DataFrame)::DataFrame
    data = DataFrame()
    for year_dt in unique(group_n[:, :Year])
        tickers = group_n[group_n[:, :Year] .== year_dt, :Ticker]
        prices_subset = prices[
            (year.(prices[:, :Date]) .== year(year_dt) + 1) .&
                (prices[:, :Ticker] .∈ Ref(tickers)), :,
        ]
        append!(data, prices_subset)
    end
    return data
end

function returns(prices::DataFrame)::DataFrame
    data = DataFrame()

    for ticker_group in groupby(prices, :Ticker)
        @assert issorted(ticker_group, :Date)

        if nrow(ticker_group) > 1
            ticker_returns = DataFrame(
                Date = ticker_group[2:end, :Date],
                Ticker = ticker_group[2:end, :Ticker],
                Close = diff(ticker_group[:, :Close]) ./ ticker_group[1:(end - 1), :Close]
            )
            append!(data, ticker_returns)
        end
    end
    return data
end

function portfolio_returns(returns::DataFrame)
    date_index = returns[:, :Date]
    years = year.(date_index)
    data = DataFrame()
    for year in unique(years)
        subset = returns[years .== year, :]
        ar = combine(groupby(subset, :Date), :Close => mean => :Close)
        append!(data, ar)
    end
    return sort(data, :Date)
end

function cum_returns(portfolio_returns::DataFrame; starting_value::Float64 = 1.0)::DataFrame
    @assert issorted(portfolio_returns, :Date)

    nanmask = isnan.(portfolio_returns[:, :Close])
    r = copy(portfolio_returns[:, :Close])
    r[nanmask] .= 0.0

    r = r .+ 1.0

    cum_r = cumprod(r)

    if starting_value == 0
        cum_r = cum_r .- 1.0
    else
        cum_r = cum_r .* starting_value
    end

    data = DataFrame(
        Date = portfolio_returns[:, :Date],
        Close = cum_r
    )

    return data
end


function get_pandas_attr(file::Parquet.File)::Dict{String, Dict{String, String}}
    meta = file.meta.key_value_metadata
    for kv in meta
        if kv.key == "PANDAS_ATTRS"
            return JSON.parse(String(kv.value))
        end
    end
    return Dict{String, Dict{String, String}}()
end


"""
copied from https://stackoverflow.com/questions/54195315/how-to-convert-a-nanosecond-precision-epoch-timestamp-to-a-datetime-in-julia
"""
function epochns_to_datetime(value::Int64)::TimeDate
    sec = value ÷ 10^9
    ms = value ÷ 10^6 - sec * 10^3
    ns = value % 10^6
    origin = TimeDate(unix2datetime(sec))
    ms = Millisecond(ms)
    ns = Nanosecond(ns)
    result = origin + ms + ns
    return result
end

function read(input::Parquet.File)::DataFrame
    table = read_parquet(input.path)
    df = DataFrame(table)
    return df
end
function compute_rank(fundamentals::DataFrame, groups::Dict{String, String})::DataFrame
    group_columns = Dict{String, Vector{String}}()
    for (col, group) in groups
        if haskey(group_columns, group)
            push!(group_columns[group], col)
        else
            group_columns[group] = [col]
        end
    end

    dim = nrow(fundamentals)
    rank = Vector{Float64}(undef, dim)
    scores = Dict(group => Vector{Float64}(undef, dim) for group in keys(group_columns))


    for (group, cols) in group_columns
        avg = mean.(eachrow(fundamentals[:, cols]))
        scores[group] = avg
    end

    data = DataFrame(
        Year = fundamentals[:, :Year],
        Ticker = fundamentals[:, :Ticker]
    )
    score = sum_scores(scores)
    for year in unique(fundamentals[:, :Year])
        year_indices = findall(y -> y == year, fundamentals[:, :Year])
        subset = score[year_indices]
        rank[year_indices] = denserank(subset, rev = true)
    end
    data[!, :Rank] = rank
    return data
end

function sum_scores(data::Dict)
    first_key = first(keys(data))
    n = length(data[first_key])

    result = zeros(eltype(data[first_key]), n)

    for (_, vec) in data
        @assert length(vec) == n "All vectors must have the same length"
        result .+= vec
    end
    return result
end

function get_fundamental_names(names::Vector{String})
    return filter(name -> name ∉ ["Year", "Ticker"], names)
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
        rt = returns(top_mirrored)
        rb = returns(bottom_mirrored)
        pt = portfolio_returns(rt)
        pb = portfolio_returns(rb)
        ct = cum_returns(pt)
        cb = cum_returns(pb)

        fx1 = last(ct[:, :Close])
        fx2 = last(cb[:, :Close])
        fx = [-fx1, fx2] # fx1 is negated to turn it into maximization
        gx = [0.0]
        hx = [0.0]
        return fx, gx, hx
    end
end

function top_n(rank::DataFrame, n::Int64; rev::Bool = false)::DataFrame
    gd = groupby(rank, :Year)
    dim = n * length(gd)
    tops = Vector{String}(undef, dim)
    years = Vector{eltype(rank[:, :Year])}(undef, dim)

    idx = 1
    for year_subset in gd
        sorted = sort(year_subset, :Rank, rev = rev)
        rows = nrow(sorted)
        if rows < n
            throw(BoundsError("n is greater than the number of rows"))
        end

        top = first(sorted[:, :Ticker], n)

        year_val = year_subset[1, :Year]

        for j in 1:n
            tops[idx] = top[j]
            years[idx] = year_val
            idx += 1
        end
    end

    data = DataFrame(
        Ticker = tops,
        Year = years
    )

    return data
end


function bottom_n(rank::DataFrame, n::Int64)
    return top_n(rank, n, rev = true)
end


function plot_portfolio(prices::DataFrame, fundamentals::DataFrame, groups::Dict{String, String})
        rank = compute_rank(fundamentals, groups)
        tn = top_n(rank, 3)
        bn = bottom_n(rank, 3)

        top_mirrored = mirror(prices, tn)
        bottom_mirrored = mirror(prices, bn)
        rt = returns(top_mirrored)
        rb = returns(bottom_mirrored)
        pt = portfolio_returns(rt)
        pb = portfolio_returns(rb)
        ct = cum_returns(pt)
        cb = cum_returns(pb)
        x = DateTime.(ct[:, :Date]) 
        y1 = ct[:, :Close]
        y2 = cb[:, :Close]
        return plot(x, [y1, y2])
end

function run(prices::DataFrame, fundamentals::DataFrame, fundamental_groups::Dict{String, String}, verbose::Bool)
    n_features = ncol(fundamentals) - 2 # Remove 2 to ignore Year and Ticker column
    lb = zeros(n_features)
    ub = ones(n_features)
    bounds = BoxConstrainedSpace(lb = lb, ub = ub)
    result = optimize(
        fitness(prices, fundamentals, fundamental_groups),
        bounds,
        NSGA2(options = Options(verbose = verbose))
    )
    optimal_mask = minimizer(result) .> THRESHOLD
    fundamental_names = get_fundamental_names(names(fundamentals))
    colnames = fundamental_names[optimal_mask]
    
    p1 = plot_portfolio(prices, fundamentals, fundamental_groups)
    savefig(p1, "notoptimized.png")
    filtered_groups = Dict(k => v for (k, v) in fundamental_groups if k in colnames)
    selected_columns = [["Year", "Ticker"]..., colnames...]
    p2 = plot_portfolio(prices, fundamentals[:, selected_columns], filtered_groups)
    savefig(p2, "optimized.png")
end


@Comonicon.main function main(prices_path::AbstractString, fundamentals_path::AbstractString; verbose::Bool = false)
    prices_input = Parquet.File(prices_path)
    prices = read(prices_input)
    prices[!, :Date] = epochns_to_datetime.(prices[:, :Date])
    fundamentals_input = Parquet.File(fundamentals_path)
    fundamentals = read(fundamentals_input)
    fundamentals[!, :Year] = epochns_to_datetime.(fundamentals[:, :Year])
    fundamentals_groups = get_pandas_attr(fundamentals_input)["columns"]

    run(prices, fundamentals, fundamentals_groups, verbose)
end
end
