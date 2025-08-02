module Utils
using DataFrames
using StatsBase
using Statistics
using TimesDates
using Dates
using Plots

export get_fundamental_names
export sum_scores, compute_rank, top_n, bottom_n
export cum_returns, returns, portfolio_returns, cum_returns_strategies
export mirror, compute_scores, safe_mean
export plot_portfolio, epochns_to_datetime

"""
    mirror(prices::DataFrame, group_n::DataFrame) -> DataFrame

Selects and concatenates rows from `prices` corresponding to the tickers and years specified in `group_n`, but for the year following each entry in `group_n`.

# Arguments
- `prices::DataFrame`: DataFrame containing price information with columns `:Date` and `:Ticker`.
- `group_n::DataFrame`: DataFrame specifying groups with columns `:Year` and `:Ticker`.

# Returns
- `DataFrame`: A new DataFrame containing all rows from `prices` where the year is one greater than the `:Year` in `group_n` and the `:Ticker` matches. The result is concatenated over all years in `group_n`.

# Example

```julia
# group_n contains tickers and years of interest
mirror(prices, group_n)
# Returns price rows for each ticker in group_n, but for the year after each group_n :Year
```
"""
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

"""
    returns(prices::Union{DataFrame, SubDataFrame}) -> DataFrame

Calculates period-over-period returns for each ticker, preserving the input dimension.
The first row for each ticker will have a `NaN` return (since no previous value exists).

# Arguments
- `prices::Union{DataFrame, SubDataFrame}`: DataFrame with columns `:Ticker`, `:Date`, and `:Close`.
  Assumes each ticker's data is sorted by `:Date`.

# Returns
- `DataFrame`: DataFrame with columns `:Ticker`, `:Date`, and `:Close`, where `:Close` is the return
  computed as `(current_close - previous_close) / previous_close`, and is `NaN` for the first row of each ticker.

# Example

```julia
prices = DataFrame(
    Ticker = ["A", "A", "B", "B", "B"],
    Date = [Date("2020-01-01"), Date("2020-01-02"), Date("2020-01-01"), Date("2020-01-02"), Date("2020-01-03")],
    Close = [100.0, 110.0, 200.0, 210.0, 220.0]
)
returns(prices)
# Output:
# 5×3 DataFrame
# Row │ Ticker  Date        Close
#     │ String  Date        Float64
#─────┼─────────────────────────────
#   1 │ A       2020-01-01  NaN
#   2 │ A       2020-01-02  0.10
#   3 │ B       2020-01-01  NaN
#   4 │ B       2020-01-02  0.05
#   5 │ B       2020-01-03  0.047619
"""
function returns(prices::Union{DataFrame, SubDataFrame})::DataFrame
    data = DataFrame()
    for ticker_group in groupby(prices, :Ticker)
        @assert issorted(ticker_group, :Date)
        n = nrow(ticker_group)
        ret = Array{Float64}(undef, n)
        ret[1] = NaN
        if n > 1
            ret[2:end] .= diff(ticker_group[:, :Close]) ./ ticker_group[1:(end - 1), :Close]
        end
        ticker_returns = DataFrame(Ticker = ticker_group[:, :Ticker], Date = ticker_group[:, :Date], Close = ret)
        append!(data, ticker_returns)
    end
    return data
end

"""
    portfolio_returns(returns::Union{DataFrame, SubDataFrame}) -> DataFrame

Calculates the mean return across all tickers for each date.

# Arguments
- `returns::Union{DataFrame, SubDataFrame}`: DataFrame containing columns `:Date` and `:Close`, where `:Close` contains return values for each ticker and date.

# Returns
- `DataFrame`: A DataFrame with columns `:Date` and `:Close`, where `:Close` is the mean return across all tickers for that date, sorted by date.

# Example

```julia
using DataFrames

returns = DataFrame(
    Ticker = ["A", "A", "B", "B", "B"],
    Date = [Date("2020-01-01"), Date("2020-01-02"), Date("2020-01-01"), Date("2020-01-02"), Date("2020-01-03")],
    Close = [NaN, 0.10, NaN, 0.05, 0.047619047619047616]
)

portfolio_returns(returns)
# Output:
# 3×2 DataFrame
# Row │ Date        Close
#     │ Date        Float64
#─────┼─────────────────────
#   1 │ 2020-01-01  NaN
#   2 │ 2020-01-02  0.075
#   3 │ 2020-01-03  0.047619047619047616
"""
function portfolio_returns(returns::Union{DataFrame, SubDataFrame})
    return sort(combine(groupby(returns, :Date), :Close => mean => :Close), :Date)
end

"""
    cum_returns(portfolio_returns::Union{DataFrame, SubDataFrame}; starting_value::Float64 = 1.0) -> DataFrame

Computes cumulative returns over time from a DataFrame of portfolio returns.

# Arguments
- `portfolio_returns::Union{DataFrame, SubDataFrame}`: DataFrame containing columns `:Date` and `:Close`, where `:Close` is the periodic return (e.g., daily or yearly), and `:Date` is sorted in ascending order.
- `starting_value::Float64 = 1.0`: Initial value for cumulative returns. If 0, the output will be the cumulative return as a growth factor (i.e., subtracting 1 at the end).

# Returns
- `DataFrame`: DataFrame with columns `:Date` and `:Close`, where `:Close` is the cumulative return at each date.

# Details
- NaN values in `:Close` are treated as zero returns.
- The cumulative return is calculated as the cumulative product of (return + 1) at each step.

# Example

```julia
using DataFrames

portfolio_returns = DataFrame(
    Date = [Date("2020-01-01"), Date("2020-01-02"), Date("2020-01-03")],
    Close = [NaN, 0.10, 0.05]
)

cum_returns(portfolio_returns)
# Output:
# 3×2 DataFrame
# Row │ Date        Close
#     │ Date        Float64
#─────┼─────────────────────
#   1 │ 2020-01-01  1.0
#   2 │ 2020-01-02  1.1
#   3 │ 2020-01-03  1.155
"""
function cum_returns(portfolio_returns::Union{DataFrame, SubDataFrame}; starting_value::Float64 = 1.0)::DataFrame
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
    cum_returns_strategies(
        returns_group::GroupedDataFrame; starting_value = 1.0
    ) -> DataFrame

Computes cumulative returns for each group (e.g., annual or strategy periods) in a `GroupedDataFrame`, chaining the final cumulative value from one group as the starting value for the next.

# Arguments
- `returns_group::GroupedDataFrame`: A grouped DataFrame, typically grouped by a period such as year, where each group contains columns `:Date` and `:Close` (return).
- `starting_value`: The initial value for the cumulative returns. Defaults to `1.0`.

# Returns
- `DataFrame`: Concatenated DataFrame of cumulative returns across all groups, with the starting value of each group set to the final cumulative value of the previous group.

# Details
- Each group's DataFrame is processed in sorted order of its keys.
- For each group, the cumulative returns are computed (using `cum_returns`), beginning with the cumulative value at the end of the previous group.
- The output DataFrame contains all dates and cumulative returns, preserving the order.

# Example

```julia
using DataFrames

returns = DataFrame(
    Year = [2020, 2020, 2021, 2021],
    Date = [Date("2020-01-01"), Date("2020-01-02"), Date("2021-01-01"), Date("2021-01-02")],
    Close = [0.05, 0.10, 0.02, 0.03]
)

returns_group = groupby(returns, :Year)

cum_returns_strategies(returns_group)
# Output:
# 4×2 DataFrame
# Row │ Date        Close
#     │ Date        Float64
#─────┼─────────────────────
#   1 │ 2020-01-01    1.05
#   2 │ 2020-01-02    1.155
#   3 │ 2021-01-01    1.1781
#   4 │ 2021-01-02    1.213443
"""
function cum_returns_strategies(
        returns_group::GroupedDataFrame; starting_value = 1.0
    )
    sorted_years = sort(collect(keys(returns_group)))

    result_df = DataFrame()
    last_value = starting_value

    for year in sorted_years
        year_df = returns_group[year]
        cum_returns_year = cum_returns(year_df; starting_value = last_value)
        append!(result_df, cum_returns_year)

        last_value = cum_returns_year[end, :Close]
    end
    return result_df
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

    transform!(top_mirrored, :Date => ByRow(date -> year(date)) => :Year)
    transform!(bottom_mirrored, :Date => ByRow(date -> year(date)) => :Year)
    rt = combine(groupby(top_mirrored, :Year), returns)
    rb = combine(groupby(bottom_mirrored, :Year), returns)
    pt = combine(groupby(rt, :Year), portfolio_returns)
    pb = combine(groupby(rb, :Year), portfolio_returns)

    ct = cum_returns_strategies(groupby(pt, :Year))
    cb = cum_returns_strategies(groupby(pb, :Year))


    x = DateTime.(ct[:, :Date])
    y1 = ct[:, :Close]
    y2 = cb[:, :Close]
    @assert length(y1) == length(y2)
    @assert length(y1) == length(x)
    p = plot(x, y1, label = "Top Portfolio")
    plot!(p, x, y2, label = "Bottom Portfolio")
    return p
end


end
