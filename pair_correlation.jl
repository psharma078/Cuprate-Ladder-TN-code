using MPI

using ITensors, ITensorMPS, HDF5, Random, TOML

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if length(ARGS) < 1
    error("Usage: julia run.jl input_file.toml")
end
params = TOML.parsefile(ARGS[1])

L = params["L"]
Nup = params["Nup"]
V = params["Vn"]

function four_pt_correlator(psi::MPS, ops::NTuple{4, String}, op_sites::Vector{NTuple{4, Int64}})
    s = siteinds(psi)
    n_ops = length(op_sites)

    partial_corr = zeros(ComplexF64, n_ops)

    for idx in rank+1:nprocs:n_ops
        s1, s2, s3, s4 = op_sites[idx]
        os = OpSum()
        os += ops[1], s1, ops[2], s2, ops[3], s3, ops[4], s4
        corr = MPO(os, s)
        partial_corr[idx] = inner(psi', corr, psi)
    end

    return  MPI.Reduce(partial_corr, +, 0, comm)
end

fname = "psi_L$(L)_Nup$(Nup)_Ndn$(Nup)_Vn$(V).h5"

#sites = siteinds("Electron",20)
#psi = randomMPS(sites,5)

f = h5open(fname,"r")
    psi = read(f,"psi",MPS)
close(f)

MPI.Barrier(comm)

if rank==0
    @show L
    @show Nup
    @show V

    file_name = "data_corr_L$(L)_Nup$(Nup)_Ndn$(Nup)_Vn$(V).h5"    
end

op1 = ("Cdagup","Cdagdn","Cdn","Cup")
op2 = ("Cdagup","Cdagdn","Cup","Cdn")
op3 = ("Cdagdn","Cdagup","Cdn","Cup")
op4 = ("Cdagdn","Cdagup","Cup","Cdn")

#=
## along rung-rung
indx = collect(1:2:2*L)
#opsites = vec([(i,i+1,j+1,j) for j in indx, i in indx])

opsites = NTuple{4, Int}[]  #non repeated indices
for i in indx
    for j in i:2:2*L
       push!(opsites,(i,i+1,j+1,j))
    end
end

uddu = @time four_pt_correlator(psi, op1, opsites)
udud = @time four_pt_correlator(psi, op2, opsites)
dudu = @time four_pt_correlator(psi, op3, opsites)
duud = @time four_pt_correlator(psi, op4, opsites)

if rank == 0
    h5write(file_name, "RR_pairing", uddu-udud-dudu+duud)
    h5write(file_name, "RR_sites", opsites)

    println("...leg-leg pair correlation...")
end
 
pop!(indx)
leg_sites = NTuple{4, Int}[]
for i in indx
    for j in i:2:2*L-2
        push!(leg_sites,(i,i+2,j+2,j))
    end
end
    
leg_sites = vec([(i,i+2,j+2,j) for j in indx, i in indx])

uddu = @time four_pt_correlator(psi, op1, leg_sites)
udud = @time four_pt_correlator(psi, op2, leg_sites)
dudu = @time four_pt_correlator(psi, op3, leg_sites)
duud = @time four_pt_correlator(psi, op4, leg_sites)

if rank == 0
    h5write(file_name, "LL_pairing", uddu-udud-dudu+duud)
    #h5write(file_name, "udud_l1", udud)
    #h5write(file_name, "dudu_l1", dudu)
    #h5write(file_name, "duud_l1", duud)
    h5write(file_name, "LL_sites", leg_sites)
    println("...pair correaltion along rung-leg...")
end

RL_sites = NTuple{4, Int}[]
for i in indx
    for j in i:2:2*L-2
        push!(RL_sites,(i,i+1,j+2,j))
    end
end

uddu = @time four_pt_correlator(psi, op1, RL_sites)
udud = @time four_pt_correlator(psi, op2, RL_sites)
dudu = @time four_pt_correlator(psi, op3, RL_sites)
duud = @time four_pt_correlator(psi, op4, RL_sites)

if rank == 0
    h5write(file_name, "RL_pairing", uddu-udud-dudu+duud)
    h5write(file_name, "RL_sites", RL_sites)
    println("...done...")
end


indx = collect(1:2:2*L-2)

LR_sites = NTuple{4, Int}[]
for i in indx
    for j in i:2:2*L-1
        push!(LR_sites,(i,i+2,j+1,j))
    end
end

uddu = @time four_pt_correlator(psi, op1, LR_sites)
udud = @time four_pt_correlator(psi, op2, LR_sites)
dudu = @time four_pt_correlator(psi, op3, LR_sites)
duud = @time four_pt_correlator(psi, op4, LR_sites)

if rank == 0
    h5write(file_name, "LR_pairing", uddu-udud-dudu+duud)
    h5write(file_name, "LR_sites", LR_sites)
    println("...done...")
end
MPI.Barrier(comm)
MPI.Finalize()
