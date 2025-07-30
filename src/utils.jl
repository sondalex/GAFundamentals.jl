module Utils
using DataFrames
using StatsBase
using Statistics
using TimesDates
using Dates
using Plots

export get_fundamental_names
export sum_scores, compute_rank, top_n, bottom_n
export cum_returns, returns, portfolio_returns
export mirror, compute_scores, safe_mean
export plot_portfolio, epochns_to_datetime

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

function safe_mean(x::Union{AbstractVector{T}, DataFrameRow})::Float64 where {T <: Union{Missing, Real}}

    valid_values = filter(!isnan, collect(skipmissing(x)))

    if isempty(valid_values)
        return NaN
    end

    return mean(valid_values)
end


function compute_scores(fundamentals::DataFrame, groups::Dict{String, String})::Dict{String, Vector{Float64}}
    group_columns = Dict{String, Vector{String}}()
    for (col, group) in groups
        if haskey(group_columns, group)
            push!(group_columns[group], col)
        else
            group_columns[group] = [col]
        end
    end

    dim = nrow(fundamentals)
    scores = Dict(
        group => Vector{Float64}(undef, dim) for group in keys(group_columns)
    )

    for (group, cols) in group_columns
        avg = safe_mean.(eachrow(fundamentals[:, cols]))
        scores[group] = avg
    end
    return scores
end

function compute_rank(fundamentals::DataFrame, groups::Dict{String, String})::DataFrame
    dim = nrow(fundamentals)
    rank = Vector{Float64}(undef, dim)
    scores = compute_scores(fundamentals, groups)
    score = sum_scores(scores)
    for year in unique(fundamentals[:, :Year])
        year_indices = findall(y -> y == year, fundamentals[:, :Year])
        subset = score[year_indices]
        rank[year_indices] = denserank(subset, rev = true)
    end
    data = DataFrame(
        Year = fundamentals[:, :Year],
        Ticker = fundamentals[:, :Ticker]
    )
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
    @assert length(y1) == length(y2)
    @assert length(y1) == length(x)
    p = plot(x, y1, label="Top Portfolio")
    plot!(p, x, y2, label="Bottom Portfolio")
    return p
end


end
