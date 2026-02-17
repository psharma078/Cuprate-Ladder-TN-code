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
f = h5open(fname,"r")
    psi = read(f,"psi",MPS)
close(f)

#sites = siteinds("Electron", 2*L)
#psi = randomMPS(sites,4)
#MPI.Barrier(comm)

if rank==0
    @show L
    @show Nup
    @show V

    file_name = "data_corr_L$(L)_Nup$(Nup)_Ndn$(Nup)_Vn$(V).h5"

    #println("...computing correlations...")

    #CdagC = @time correlation_matrix(psi, "Cdagup", "Cup")
    #SziSzj = @time correlation_matrix(psi, "Sz", "Sz")
    #ninj = @time correlation_matrix(psi, "Ntot", "Ntot")

    #h5write(file_name, "CdagC", CdagC)
    #h5write(file_name, "ninj", ninj)
    #h5write(file_name, "SziSzj", SziSzj)

    println("...computing pair correlations...")

end

indx = collect(1:2:2*L)
opsites = vec([(i,i+1,j+1,j) for j in indx, i in indx])

op1 = ("Cdagup","Cdagdn","Cdn","Cup")
op2 = ("Cdagup","Cdagdn","Cup","Cdn")
op3 = ("Cdagdn","Cdagup","Cdn","Cup")
op4 = ("Cdagdn","Cdagup","Cup","Cdn")

uddu = @time four_pt_correlator(psi, op1, opsites)
udud = @time four_pt_correlator(psi, op2, opsites)
dudu = @time four_pt_correlator(psi, op3, opsites)
duud = @time four_pt_correlator(psi, op4, opsites)

if rank == 0
    h5write(file_name, "uddu", uddu)
    h5write(file_name, "udud", udud)
    h5write(file_name, "dudu", dudu)
    h5write(file_name, "duud", duud)

    println("...pair correlation along leg 1...")
end

opsites = vec([(L-1,L,i+2,i) for i in L+2:4:2*L-2])

uddu = @time four_pt_correlator(psi, op1, opsites)
udud = @time four_pt_correlator(psi, op2, opsites)
dudu = @time four_pt_correlator(psi, op3, opsites)
duud = @time four_pt_correlator(psi, op4, opsites)

if rank == 0
    h5write(file_name, "uddu_l1", uddu)
    h5write(file_name, "udud_l1", udud)
    h5write(file_name, "dudu_l1", dudu)
    h5write(file_name, "duud_l1", duud)
    @show uddu
    println("...pair correaltion along leg 2...")
end

opsites = vec([(L-1,L,i+2,i) for i in L+1:4:2*L-3])

uddu = @time four_pt_correlator(psi, op1, opsites)
udud = @time four_pt_correlator(psi, op2, opsites)
dudu = @time four_pt_correlator(psi, op3, opsites)
duud = @time four_pt_correlator(psi, op4, opsites)

if rank == 0
    h5write(file_name, "uddu_l2", uddu)
    h5write(file_name, "udud_l2", udud)
    h5write(file_name, "dudu_l2", dudu)
    h5write(file_name, "duud_l2", duud)
    @show uddu
    println("...done...")
end                              
