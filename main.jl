import Comonicon
using Parquet
using DataFrames
using Metaheuristics


function read(input::AbstractString)::DataFrame
    df = DataFrame(read_parquet(input)) 
    return df
end



function optimize(prices::DataFrame, fundamentals::DataFrame, verbose::bool)
    println(prices)
    println(fundamentals)
end


@Comonicon.main function main(prices_path::String, fundamentals_path::String; verbose::Bool=false)
    prices = read(prices_path)
    fundamentals = read(fundamentals_path)

    optimize(prices, fundamentals, verbose)

end


