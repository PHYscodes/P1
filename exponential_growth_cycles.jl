#!/usr/bin/env julia
using Pkg; try Pkg.activate("/home/axs2906/jvenv") catch; end
ENV["GKSwstype"] = "nul"
ENV["PLOTS_DEFAULT_BACKEND"] = "gr"
using Random
using Statistics
using Plots
using Distributions
using StatsBase
using Base.Threads

# -----------------------------
# Parameters
# -----------------------------
const POP_SIZE = 1000                         # Carrying capacity (maximum population size per cycle)
const INITIAL_FRACTION = 0.10                 # Fraction of POP_SIZE used as bottleneck size
const INITIAL_SIZE = Int(round(POP_SIZE * INITIAL_FRACTION))
const TIME_STEP_DT = 1.0                      # Discrete time step for exponential growth
const MAX_CYCLES = 30                         # Number of growth-bottleneck cycles per simulation
const MUTATION_RATES = [10.0, 1.0, 0.01, 0.0001]
const REPEATS = 1                             # Number of independent replicates per mutation rate
const FC = 0.75                               # Parameter controlling intrinsic fitness range

# -----------------------------
# Utility
# -----------------------------
function compute_adjusted_growth_rate(
    type_id::Int,
    type_frequency::Float64,
    intrinsic_fitness::Dict{Int,Float64},
    ecological_fitness::Dict{Int,Float64}
)::Float64
    # Same selection logic as the original model, reinterpreted as a growth rate
    return type_frequency * (intrinsic_fitness[type_id] - ecological_fitness[type_id]) + ecological_fitness[type_id]
end

function to_frequency_map(population::Dict{Int,Int})::Dict{Int,Float64}
    total_count = sum(values(population))
    if total_count == 0
        return Dict{Int,Float64}()
    end
    result = Dict{Int,Float64}()
    for (type_id, count) in population
        result[type_id] = count / total_count
    end
    return result
end

# -----------------------------
# Simulation: Exponential growth with periodic bottlenecks
# -----------------------------
function simulate_growth_cycles(rng::AbstractRNG, mutation_rate::Float64)
    # State
    population = Dict(1 => INITIAL_SIZE)
    intrinsic_fitness = Dict(1 => 1.0)         # Same convention as original code for the ancestor
    ecological_fitness = Dict(1 => 1.0)

    mutant_properties = Dict{Int,Tuple{Float64,Float64}}()  # type_id => (fi, fe)
    next_mutant_type = 2

    # Time series for plotting
    total_population_over_time = Int[]
    cycle_end_steps = Int[]

    # Δp tracking across cycles
    delta_p_values = Float64[]
    start_of_cycle_frequencies = Dict(1 => 1.0)  # Frequencies at the start of the current cycle (post-bottleneck)

    cycles_completed = 0
    time_step = 0

    while cycles_completed < MAX_CYCLES
        time_step += 1

        # Early exit if population extinct (should not generally happen with these parameters)
        if isempty(population)
            break
        end

        # Current total population before mutation and growth
        total_count = sum(values(population))

        # -----------------
        # Mutation step
        # -----------------
        # Draw new mutants proportional to current population size
        new_mutants = rand(rng, Poisson(mutation_rate * total_count))
        if new_mutants > 0
            types = collect(keys(population))
            counts = [population[t] for t in types]
            if !isempty(types)
                for _ in 1:new_mutants
                    # Choose parent weighted by counts
                    parent_type = sample(rng, types, Weights(counts))
                    # Update parent count
                    population[parent_type] -= 1
                    if population[parent_type] == 0
                        delete!(population, parent_type)
                        # Also update local cached arrays if we want balanced sampling in later mutant draws
                        # But for simplicity, we will rebuild types/counts next mutant; cost is negligible for small new_mutants
                        types = collect(keys(population))
                        counts = [population[t] for t in types]
                        if isempty(types)
                            break
                        end
                    end

                    # Create or reuse mutant type based on (fi, fe)
                    fi = rand(rng) * (1 - FC)
                    fe = rand(rng)

                    mutant_type = 0
                    for (existing_type, properties) in mutant_properties
                        if isapprox(properties[1], fi) && isapprox(properties[2], fe)
                            mutant_type = existing_type
                            break
                        end
                    end

                    if mutant_type == 0
                        mutant_type = next_mutant_type
                        next_mutant_type += 1
                        intrinsic_fitness[mutant_type] = fi
                        ecological_fitness[mutant_type] = fe
                        mutant_properties[mutant_type] = (fi, fe)
                    end

                    population[mutant_type] = get(population, mutant_type, 0) + 1
                    # Refresh arrays for weighted sampling consistency
                    types = collect(keys(population))
                    counts = [population[t] for t in types]
                end
            end
        end

        # -----------------
        # Exponential growth step: n_i(t+dt) = n_i(t) * exp(g_i * dt)
        # -----------------
        if isempty(population)
            break
        end
        types = collect(keys(population))
        counts = [population[t] for t in types]
        total_count = sum(counts)
        frequencies = [c / total_count for c in counts]

        new_counts = Int[]
        for (idx, type_id) in enumerate(types)
            g = compute_adjusted_growth_rate(type_id, frequencies[idx], intrinsic_fitness, ecological_fitness)
            # Deterministic exponential growth, rounded to Int
            grown = Int(round(counts[idx] * exp(g * TIME_STEP_DT)))
            push!(new_counts, max(grown, 0))
        end

        # Rebuild the population dictionary
        population = Dict(t => c for (t, c) in zip(types, new_counts) if c > 0)

        # Track total population
        push!(total_population_over_time, sum(new_counts))

        # -----------------
        # Bottleneck when reaching carrying capacity
        # -----------------
        if sum(new_counts) >= POP_SIZE
            # Sample back down to INITIAL_SIZE to start the next cycle
            types2 = collect(keys(population))
            counts2 = [population[t] for t in types2]
            total2 = sum(counts2)
            if total2 == 0
                break
            end
            freqs2 = [c / total2 for c in counts2]
            sampled_counts = rand(rng, Multinomial(INITIAL_SIZE, freqs2))
            new_population = Dict(t => s for (t, s) in zip(types2, sampled_counts) if s > 0)

            # Compute Δp for this cycle: p' - p
            end_of_cycle_frequencies = to_frequency_map(new_population)  # these become the next cycle's start freqs
            all_types = union(keys(start_of_cycle_frequencies), keys(end_of_cycle_frequencies))
            for type_id in all_types
                p_prev = get(start_of_cycle_frequencies, type_id, 0.0)
                p_next = get(end_of_cycle_frequencies, type_id, 0.0)
                push!(delta_p_values, p_next - p_prev)
            end

            # Prepare for next cycle
            population = new_population
            start_of_cycle_frequencies = end_of_cycle_frequencies

            cycles_completed += 1
            push!(cycle_end_steps, time_step)
        end
    end

    return delta_p_values, total_population_over_time, cycle_end_steps
end

# -----------------------------
# Runner per mutation rate
# -----------------------------
function run_simulation(mutation_rate::Float64)
    println("Starting exponential growth cycles for μ=$mutation_rate with $(nthreads()) threads...")

    seeds = rand(MersenneTwister(42), UInt32, REPEATS)

    delta_p_per_thread = [Float64[] for _ in 1:nthreads()]
    ts_reports = Vector{Vector{Int}}(undef, REPEATS)
    cb_reports = Vector{Vector{Int}}(undef, REPEATS)

    progress = 0
    progress_lock = ReentrantLock()

    @threads for i in 1:REPEATS
        tid = threadid()
        rng = MersenneTwister(seeds[i])
        dps, ts, cbs = simulate_growth_cycles(rng, mutation_rate)
        append!(delta_p_per_thread[tid], dps)
        ts_reports[i] = ts
        cb_reports[i] = cbs

        lock(progress_lock) do
            progress += 1
            if progress % 20 == 0 || progress == REPEATS
                println("Progress for μ=$mutation_rate: $(progress)/$(REPEATS) simulations completed")
            end
        end
    end

    delta_p_all = vcat(delta_p_per_thread...)
    mean_delta_p = isempty(delta_p_all) ? 0.0 : mean(delta_p_all)
    var_delta_p = length(delta_p_all) <= 1 ? 0.0 : var(delta_p_all)

    # Use the first replicate's time series for visualization
    ts_example = ts_reports[1]
    cb_example = cb_reports[1]

    return ts_example, cb_example, mean_delta_p, var_delta_p
end

# -----------------------------
# Main entry
# -----------------------------
function main()
    colors = [:red, :blue, :green, :purple, :orange, :teal]

    # Collect stats and example time series
    mean_dps = Float64[]
    var_dps = Float64[]

    plt_ts = plot(
        layout=(length(MUTATION_RATES), 1),
        size=(900, 260 * length(MUTATION_RATES)),
        legend=false,
        link=:x
    )

    for (i, μ) in enumerate(MUTATION_RATES)
        ts, cb, mean_dp, var_dp = run_simulation(μ)
        push!(mean_dps, mean_dp)
        push!(var_dps, var_dp)

        # Time series subplot for this μ
        if isempty(ts)
            plot!(plt_ts, subplot=i, title="μ=$μ (no data)")
        else
            plot!(
                plt_ts,
                1:length(ts),
                ts,
                color=colors[mod1(i, length(colors))],
                xlabel="time step",
                ylabel="Total population",
                title="Exponential growth with bottlenecks (μ=$μ)",
                subplot=i
            )
            # Mark cycle endings
            if !isempty(cb)
                for b in cb
                    vline!(plt_ts, [b], color=:black, linestyle=:dash, alpha=0.4, subplot=i)
                end
            end
        end
    end

    # Plot mean Δp with variance (as yerror = std)
    std_dps = [sqrt(v) for v in var_dps]
    plt_stats = plot(
        MUTATION_RATES,
        mean_dps,
        yerror=std_dps,
        xscale=:log10,
        seriestype=:scatter,
        markershape=:circle,
        color=:black,
        xlabel="Mutation rate (μ)",
        ylabel="Mean Δp (±1 SD)",
        title="Mean and variability of Δp across cycles",
        legend=false,
        size=(800, 500)
    )

    # Save and display
    println("Displaying plots...")
    display(plt_ts)
    display(plt_stats)

    println("Saving plots...")
    savefig(plt_ts, "population_timeseries_with_cycles.png")
    savefig(plt_stats, "delta_p_stats_by_mu.png")
    println("Plots saved: 'population_timeseries_with_cycles.png', 'delta_p_stats_by_mu.png'")

    println("Summary statistics (per μ):")
    for (μ, m, v) in zip(MUTATION_RATES, mean_dps, var_dps)
        println("  μ=$(μ): mean Δp=$(round(m, digits=6)), var Δp=$(round(v, digits=6))")
    end

    println("Simulation complete!")
end

main()