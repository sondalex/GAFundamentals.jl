using GAFundamentals.Utils
using DataFrames
using Test
using Dates

using GAFundamentals.Utils
using DataFrames
using Test
using Statistics

function dataframes_equal(
        df1::DataFrame, df2::DataFrame;
        check_column_order::Bool = true,
        check_row_order::Bool = true,
        rtol::Float64 = 1.0e-5
    )
    if check_column_order
        if names(df1) != names(df2)
            return false
        end
    elseif Set(names(df1)) != Set(names(df2))
        return false
    end

    if size(df1) != size(df2)
        return false
    end

    if !check_row_order
        cols = [name for name in names(df1) if eltype(df1[!, name]) <: Union{Number, String, Symbol}]
        if !isempty(cols)
            df1 = sort(df1, cols)
            df2 = sort(df2, cols)
        end
    end

    for col in names(df1)
        values1 = df1[!, col]
        values2 = df2[!, col]

        if eltype(values1) <: Number && eltype(values2) <: Number
            for (v1, v2) in zip(values1, values2)
                if isnan(v1) && isnan(v2)
                    continue
                elseif isnan(v1) || isnan(v2)
                    return false
                elseif !isapprox(v1, v2, rtol = rtol)
                    return false
                end
            end
        elseif values1 != values2
            return false
        end
    end

    return true
end

@testset "safe_mean tests" begin
    @testset "Normal numeric values" begin
        @test safe_mean([1.0, 2.0, 3.0]) ≈ 2.0
        @test safe_mean([1, 2, 3, 4]) ≈ 2.5
        @test safe_mean([-1.5, 2.5]) ≈ 0.5
    end

    @testset "NaN values" begin
        @test safe_mean([1.0, NaN, 3.0]) ≈ 2.0
        @test safe_mean([NaN, 2.0, 3.0, NaN]) ≈ 2.5
        @test safe_mean([NaN, 10.0]) ≈ 10.0
    end

    @testset "All NaN values" begin
        @test isnan(safe_mean([NaN, NaN, NaN]))
    end

    @testset "Empty collection" begin
        @test isnan(safe_mean(Float64[]))
        @test isnan(safe_mean(Int[]))
    end

    @testset "Missing values" begin
        @test safe_mean([1.0, missing, 3.0]) ≈ 2.0
        @test safe_mean([missing, 10.0, missing]) ≈ 10.0
        @test isnan(safe_mean([missing, missing, missing]))
    end

    @testset "Mixed NaN and missing" begin
        @test safe_mean([1.0, NaN, missing, 3.0]) ≈ 2.0
        @test isnan(safe_mean([NaN, missing, NaN, missing]))
    end

    @testset "Different vector types" begin
        @test safe_mean(view([1.0, 2.0, 3.0, 4.0], 1:3)) ≈ 2.0
        @test safe_mean(reshape([1.0, 2.0, 3.0], 3, 1)[:]) ≈ 2.0
    end

    @testset "Edge cases" begin
        @test safe_mean([42.0]) ≈ 42.0
        @test safe_mean([42]) == 42.0

        @test safe_mean([1.0e10, 2.0e10, 3.0e10]) ≈ 2.0e10

        @test safe_mean([1.0e-10, 2.0e-10, 3.0e-10]) ≈ 2.0e-10

        @test safe_mean([1, 2.0, 3]) ≈ 2.0
    end

    @testset "Integration with compute_scores" begin
        fundamentals = DataFrame(
            Year = [2020, 2020],
            Ticker = ["AAPL", "MSFT"],
            Value1 = [10.0, NaN],
            Value2 = [20.0, 15.0]
        )

        groups = Dict(
            "Value1" => "Group1",
            "Value2" => "Group1"
        )

        scores = compute_scores(fundamentals, groups)

        @test scores["Group1"] ≈ [
            (10.0 + 20.0) / 2,
            15.0,
        ]
    end

    @testset "Integration with empty columns" begin
        fundamentals = DataFrame(
            Year = [2020, 2020],
            Ticker = ["AAPL", "MSFT"]
        )

        empty_groups = Dict("NonExistentColumn" => "EmptyGroup")

        @test_throws Exception compute_scores(fundamentals, empty_groups)
    end
end


@testset "compute_scores tests" begin
    @testset "Basic functionality" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020, 2021, 2021, 2021],
            Ticker = ["AAPL", "MSFT", "GOOG", "AAPL", "MSFT", "GOOG"],
            Value1 = [10.0, 5.0, 7.5, 12.0, 8.0, 9.0],
            Value2 = [20.0, 15.0, 18.0, 22.0, 17.0, 20.0],
            Metric1 = [0.8, 0.4, 0.6, 0.9, 0.7, 0.5]
        )

        groups = Dict(
            "Value1" => "Group1",
            "Value2" => "Group1",
            "Metric1" => "Group2"
        )

        scores = compute_scores(fundamentals, groups)

        @test length(keys(scores)) == 2
        @test haskey(scores, "Group1")
        @test haskey(scores, "Group2")
        @test length(scores["Group1"]) == 6
        @test length(scores["Group2"]) == 6

        @test scores["Group1"] ≈ [
            (10.0 + 20.0) / 2,  # AAPL 2020,
            (5.0 + 15.0) / 2,   # MSFT 2020
            (7.5 + 18.0) / 2,   # GOOG 2020
            (12.0 + 22.0) / 2,  # AAPL 2021
            (8.0 + 17.0) / 2,   # MSFT 2021
            (9.0 + 20.0) / 2,   # GOOG 2021
        ]
        @test scores["Group2"] ≈ [
            0.8,  # AAPL 2020,
            0.4,  # MSFT 2020
            0.6,  # GOOG 2020
            0.9,  # AAPL 2021
            0.7, # MSFT 2021
            0.5,  # GOOG 2021
        ]
    end

    @testset "Empty groups dictionary" begin
        fundamentals = DataFrame(
            Year = [2020, 2021],
            Ticker = ["AAPL", "MSFT"],
            Value = [10.0, 15.0]
        )

        groups = Dict{String, String}()

        scores = compute_scores(fundamentals, groups)
        @test isempty(scores)
    end

    @testset "Single group with multiple columns" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Metric1 = [0.9, 0.5, 0.7],
            Metric2 = [0.8, 0.6, 0.4],
            Metric3 = [0.7, 0.8, 0.6]
        )

        groups = Dict(
            "Metric1" => "Group1",
            "Metric2" => "Group1",
            "Metric3" => "Group1"
        )

        scores = compute_scores(fundamentals, groups)

        @test scores["Group1"] ≈ [
            (0.9 + 0.8 + 0.7) / 3, # AAPL average
            (0.5 + 0.6 + 0.8) / 3, # MSFT average
            (0.7 + 0.4 + 0.6) / 3,  # GOOG average
        ]
    end

    @testset "Multiple groups" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            ValMetric1 = [10.0, 5.0, 8.0],
            ValMetric2 = [20.0, 15.0, 18.0],
            GrowthMetric1 = [0.9, 0.5, 0.7],
            GrowthMetric2 = [0.8, 0.6, 0.4]
        )

        groups = Dict(
            "ValMetric1" => "Valuation",
            "ValMetric2" => "Valuation",
            "GrowthMetric1" => "Growth",
            "GrowthMetric2" => "Growth"
        )

        scores = compute_scores(fundamentals, groups)

        @test scores["Valuation"] ≈ [
            (10.0 + 20.0) / 2,  # AAPL valuation
            (5.0 + 15.0) / 2,   # MSFT valuation
            (8.0 + 18.0) / 2,   # GOOG valuation
        ]

        @test scores["Growth"] ≈ [
            (0.9 + 0.8) / 2,  # AAPL growth
            (0.5 + 0.6) / 2,  # MSFT growth
            (0.7 + 0.4) / 2,  # GOOG growth
        ]
    end

    @testset "Handling NaN values" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Metric1 = [0.9, NaN, 0.7],   # MSFT has NaN in Metric1
            Metric2 = [0.8, 0.6, 0.4],
            Metric3 = [0.7, 0.8, NaN]    # GOOG has NaN in Metric3
        )

        groups = Dict(
            "Metric1" => "Group1",
            "Metric2" => "Group1",
            "Metric3" => "Group1"
        )

        scores = compute_scores(fundamentals, groups)

        @test scores["Group1"] ≈ [
            (0.9 + 0.8 + 0.7) / 3,  # AAPL: all values available
            (0.6 + 0.8) / 2,       # MSFT: ignoring NaN
            (0.7 + 0.4) / 2,       # GOOG: ignoring NaN
        ]
        fundamentals2 = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            ValMetric1 = [10.0, 5.0, NaN],
            ValMetric2 = [20.0, NaN, NaN],  # GOOG has all NaNs in Valuation group
            GrowthMetric1 = [0.9, 0.5, 0.7],
            GrowthMetric2 = [0.8, 0.6, 0.4]
        )

        groups2 = Dict(
            "ValMetric1" => "Valuation",
            "ValMetric2" => "Valuation",
            "GrowthMetric1" => "Growth",
            "GrowthMetric2" => "Growth"
        )

        scores2 = compute_scores(fundamentals2, groups2)

        @test isequal(
            scores2["Valuation"], [
                (10.0 + 20.0) / 2,  # AAPL: normal case
                5.0,               # MSFT: only one value available
                NaN,
            ]
        )
        @test scores2["Growth"] ≈ [
            (0.9 + 0.8) / 2,  # AAPL
            (0.5 + 0.6) / 2,  # MSFT
            (0.7 + 0.4) / 2,  # GOOG
        ]
    end
end

@testset "compute_rank tests" begin
    @testset "Basic functionality" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020, 2021, 2021, 2021],
            Ticker = ["AAPL", "MSFT", "GOOG", "AAPL", "MSFT", "GOOG"],
            Value1 = [10.0, 5.0, 7.5, 12.0, 8.0, 9.0],
            Value2 = [20.0, 15.0, 18.0, 22.0, 17.0, 20.0],
            Metric1 = [0.8, 0.4, 0.6, 0.9, 0.7, 0.5]
        )

        groups = Dict(
            "Value1" => "Group1",
            "Value2" => "Group1",
            "Metric1" => "Group2"
        )

        result = compute_rank(fundamentals, groups)

        expected_result = DataFrame(
            Year = [2020, 2020, 2020, 2021, 2021, 2021],
            Ticker = ["AAPL", "MSFT", "GOOG", "AAPL", "MSFT", "GOOG"],
            Rank = [1.0, 3.0, 2.0, 1.0, 3.0, 2.0]
        )
        @test dataframes_equal(result, expected_result)

    end

    @testset "Single group with multiple columns" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Metric1 = [0.9, 0.5, 0.7],
            Metric2 = [0.8, 0.6, 0.4],
            Metric3 = [0.7, 0.8, 0.6]
        )

        groups = Dict(
            "Metric1" => "Group1",
            "Metric2" => "Group1",
            "Metric3" => "Group1"
        )

        result = compute_rank(fundamentals, groups)

        expected_result = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Rank = [1.0, 2.0, 3.0]
        )
        @test dataframes_equal(result, expected_result)
    end

    @testset "Multiple groups" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            ValMetric1 = [10.0, 5.0, 8.0],
            ValMetric2 = [20.0, 15.0, 18.0],
            GrowthMetric1 = [0.9, 0.5, 0.7],
            GrowthMetric2 = [0.8, 0.6, 0.4]
        )

        groups = Dict(
            "ValMetric1" => "Valuation",
            "ValMetric2" => "Valuation",
            "GrowthMetric1" => "Growth",
            "GrowthMetric2" => "Growth"
        )

        result = compute_rank(fundamentals, groups)

        expected_result = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Rank = [1.0, 3.0, 2.0]
        )
        @test dataframes_equal(result, expected_result)
    end

    @testset "Handling NaN values" begin
        fundamentals = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Metric1 = [0.9, NaN, 0.7],   # MSFT has NaN in Metric1
            Metric2 = [0.8, 0.6, 0.4],
            Metric3 = [0.7, 0.8, NaN]    # GOOG has NaN in Metric3
        )

        groups = Dict(
            "Metric1" => "Group1",
            "Metric2" => "Group1",
            "Metric3" => "Group1"
        )

        result = compute_rank(fundamentals, groups)

        expected_result = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Rank = [1.0, 2.0, 3.0]
        )
        @test dataframes_equal(result, expected_result)

        fundamentals2 = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            ValMetric1 = [10.0, 5.0, NaN],
            ValMetric2 = [20.0, NaN, 3.0],
            GrowthMetric1 = [0.9, 0.5, 0.7],
            GrowthMetric2 = [0.8, 0.6, 0.4]
        )

        groups2 = Dict(
            "ValMetric1" => "Valuation",
            "ValMetric2" => "Valuation",
            "GrowthMetric1" => "Growth",
            "GrowthMetric2" => "Growth"
        )

        result2 = compute_rank(fundamentals2, groups2)

        expected_result2 = DataFrame(
            Year = [2020, 2020, 2020],
            Ticker = ["AAPL", "MSFT", "GOOG"],
            Rank = [1.0, 2.0, 3.0]
        )
        @test dataframes_equal(result2, expected_result2)
    end
end

@testset "mirror" begin
    prices = DataFrame(
        Date = [
            Date("2020-01-01"),
            Date("2020-01-01"),
            Date("2021-01-01"),
            Date("2021-01-01"),
            Date("2022-01-01"),
            Date("2022-01-01"),
        ],
        Ticker = ["A", "B", "B", "A", "A", "B"],
        Close = [100, 105, 110, 120, 130, 140]
    )

    group_n = DataFrame(
        Year = [Date("2020-01-01"), Date("2021-01-01")],
        Ticker = ["A", "B"]
    )

    result = mirror(prices, group_n)

    expected = DataFrame(
        Date = [
            Date("2021-01-01"),
            Date("2022-01-01"),
        ],
        Ticker = ["A", "B"],
        Close = [120, 140]
    )

    @test isequal(result, expected)
end

@testset "returns" begin
    prices = DataFrame(
        Ticker = ["A", "A", "B", "B", "B"],
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-03"),
        ],
        Close = [100.0, 110.0, 200.0, 210.0, 220.0]
    )

    result = returns(prices)

    expected = DataFrame(
        Ticker = ["A", "A", "B", "B", "B"],
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-03"),
        ],

        Close = [
            NaN,
            ((110.0 - 100.0) / 100.0),
            NaN,
            ((210.0 - 200.0) / 200.0),
            ((220.0 - 210.0) / 210.0),
        ]
    )

    @test isequal(result, expected)
end

@testset "portfolio_returns computes mean return per date" begin
    returns = DataFrame(
        Ticker = ["A", "A", "B", "B", "B"],
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-03"),
        ],

        Close = [
            NaN,
            ((110.0 - 100.0) / 100.0),
            NaN,
            ((210.0 - 200.0) / 200.0),
            ((220.0 - 210.0) / 210.0),
        ]

    )
    result = portfolio_returns(returns)
    expected = DataFrame(
        Date = [Date("2020-01-01"), Date("2020-01-02"), Date("2020-01-03")],
        Close = [
            NaN,
            (((110.0 - 100.0) / 100.0) + ((210.0 - 200.0) / 200.0)) / 2,
            ((220.0 - 210.0) / 210.0),
        ]
    )
    @test isequal(result, expected)
end

@testset "cum_returns computes cumulative returns with NaN handling and starting value" begin
    portfolio_returns = DataFrame(
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-03"),
        ],

        Close = [
            NaN,
            0.1,
            0.05,
        ]
    )

    # Default starting_value = 1.0
    result_default = cum_returns(portfolio_returns)
    expected_default = DataFrame(
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-03"),
        ],
        Close = [
            1.0,
            1.0 * (1.0 + 0.1),
            1.0 * (1.0 + 0.1) * (1.0 + 0.05),
        ]
    )
    @test dataframes_equal(result_default, expected_default)

    # starting_value = 0
    result_zero = cum_returns(portfolio_returns; starting_value = 0.0)
    expected_zero = DataFrame(
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2020-01-03"),
        ],
        Close = [
            0.0,
            ((1.0) * (1.0 + 0.1)) - 1,
            (1.0 * (1.0 + 0.1) * (1.0 + 0.05)) - 1.0,
        ]
    )
    @test dataframes_equal(result_zero, expected_zero)
end

@testset "cum_returns_strategies chains cumulative returns across groups, including NaN case" begin
    returns = DataFrame(
        Year = [
            2020,
            2020,
            2021,
            2021,
            2021,
        ],
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2021-01-01"),
            Date("2021-01-02"),
            Date("2021-01-03"),
        ],
        Close = [
            0.05,
            NaN,
            0.02,
            0.03,
            NaN,
        ]
    )

    returns_group = groupby(returns, :Year)
    result = cum_returns_strategies(returns_group)

    # Calculations:
    # 2020: [1.05, 1.05] (since NaN is treated as 0 in cum_returns)
    # 2021: [1.071, 1.10313, 1.10313]
    newstarting_value = (1.0 + 0.05)
    expected = DataFrame(
        Date = [
            Date("2020-01-01"),
            Date("2020-01-02"),
            Date("2021-01-01"),
            Date("2021-01-02"),
            Date("2021-01-03"),
        ],
        Close = [
            (1.0 + 0.05),
            (1.0 + 0.05),
            newstarting_value * (1.0 + 0.02),
            newstarting_value * (1.0 + 0.02) * (1.0 + 0.03),
            newstarting_value * (1.0 + 0.02) * (1.0 + 0.03),
        ]
    )

    @test result.Date == expected.Date
    @test all(isapprox.(result.Close, expected.Close; atol = 1.0e-8))
end

@testset "top_n tests" begin
    df = DataFrame(
        Ticker = ["A", "B", "C", "D", "E", "F"], 
        Year = [2020, 2020, 2020, 2021, 2021, 2021], 
        Rank = [2, 1, 3, 3, 1, 2]
    )

    result = top_n(df, 2)
    @test result == DataFrame(
        Ticker = ["B", "A", "E", "F"], 
        Year = [2020, 2020, 2021, 2021]
    )

    result_rev = top_n(df, 2; rev = true)
    @test result_rev == DataFrame(
        Ticker = ["C", "A", "D", "F"],
        Year = [2020, 2020, 2021, 2021]
    )

    # Test n > rows: should throw BoundsError
    df_small = DataFrame(
        Ticker = ["X", "Y"],
        Year = [2022, 2022],
        Rank = [1, 2]
    )
    @test_throws BoundsError top_n(df_small, 3)

    # Test correct year extraction
    result_years = top_n(df, 1)
    @test result_years == DataFrame(
        Ticker = ["B", "E"],
        Year = [2020, 2021]
    )
end
