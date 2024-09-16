using HDF5

function run_impact(name; lower_bound=true)
    try 
        cmd = `$(@__DIR__)/IMPaCT/$name/docker_run.sh`

        stdout = read(cmd, String)

        abstraction_output, certification_output = split(stdout, "Finding control policy")

        # Find memory usage
        mem = 0.0
        for m in eachmatch(r"Approximate memory required if stored: (\d+\.\d+)(Mb|Kb)", abstraction_output)
            prob_mem = parse(Float64, m.captures[1])
            mem_unit = m.captures[2]

            if mem_unit == "Kb"
                prob_mem /= 1000
            end

            mem += prob_mem
        end

        # Find abstraction time
        abstraction_time = 0.0
        for m in eachmatch(r"Execution time: (\d+\.\d+) seconds", abstraction_output)
            t = parse(Float64, m.captures[1])
            abstraction_time += t
        end

        # Find certification time
        certification_time = 0.0
        for m in eachmatch(r"Execution time: (\d+\.\d+) seconds", certification_output)
            t = parse(Float64, m.captures[1])
            certification_time += t
        end

        # Find value function
        file = "$(@__DIR__)/IMPaCT/ex_2Drobot-R-U/controller.h5"
        table = "dataset"
        data = h5read(file, table)
        V = if lower_bound
            data[:, end - 1]
        else
            data[:, end]
        end

        # Cleanup
        rm("$(@__DIR__)/IMPaCT/$name/controller.h5", force=true)

        return (oom=false, abstraction_time=abstraction_time, certification_time=certification_time, prob_mem=mem, V=V)
    catch e
        if isa(e, ProcessFailedException)
            @error "IMPACT failed with exit code $(e.termsignal) for $name"
            
            return (oom=true, abstraction_time=NaN, certification_time=NaN, prob_mem=NaN, V=NaN)
        else
            rethrow(e)
        end
    end
end