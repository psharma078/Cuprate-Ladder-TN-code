using ITensors, ITensorMPS, HDF5, Random

function save_mps(fname, psi)
    f = h5open(fname,"w")
        write(f,"psi",psi)
    close(f)
end

function two_pt_correlator(psi::MPS)
    ss = siteinds(psi)
    ops = ("Cdagup", "Cup")
    Lo = div(length(psi),2)
    @show Lo
    correlation = zeros(Float64, 2*Lo)
    for i in 1:length(ss)
        s1, s2 = i, Lo

        op = OpSum()
        op += ops[1], s1, ops[2], s2
        corr_mpo = MPO(op, ss)
        corr_2p = inner(psi', corr_mpo, psi)
        #println(corr_2p)
        correlation[i] = corr_2p
    end
    return correlation
end


##calculates ⟨GS|nj(t) nL(0)|GS⟩ on 1D ladder of length L
function exHubbard_nqw(L::Int64, tleg::Float64, trung::Float64, tdiag::Float64, U::Float64,
                        V::Float64, Nup::Int64, Ndn::Int64)
    N = 2*L

    #nearest neighbor and next nn bonds in ladder
    sites_l1 = [2*n-1 for n in 1:L]
    sites_l2 = [2*n for n in 1:L]

    bonds_l1 = [(sites_l1[i], sites_l1[i+1]) for i in 1:L-1]  #NN bonds in leg1
    bonds_l2 = [(sites_l2[i], sites_l2[i+1]) for i in 1:L-1]  #NN bonds in leg2
    bonds_rung = [(sites_l1[i], sites_l2[i]) for i in 1:L]    #NN bonds along rung
    bonds_diag = vcat([(bonds_l1[i][1], bonds_l2[i][2]) for i in 1:L-1],
                [(bonds_l2[i][1], bonds_l1[i][2]) for i in 1:L-1]) #NNN bonds along diagonal


    os = OpSum()
    #hopping terms along leg1 and leg2
    for (s1,s2) in vcat(bonds_l1, bonds_l2)
        os += -tleg, "Cdagup", s1, "Cup", s2
        os += -tleg, "Cdagup", s2, "Cup", s1
        os += -tleg, "Cdagdn", s1, "Cdn", s2
        os += -tleg, "Cdagdn", s2, "Cdn", s1
    end
    #hopping terms along rung
    for (s1,s2) in bonds_rung
        os += -trung, "Cdagup", s1, "Cup", s2
        os += -trung, "Cdagup", s2, "Cup", s1
        os += -trung, "Cdagdn", s1, "Cdn", s2
        os += -trung, "Cdagdn", s2, "Cdn", s1
    end
    #hopping terms along diagonals
    for (s1,s2) in bonds_diag
        os += -tdiag, "Cdagup", s1, "Cup", s2
        os += -tdiag, "Cdagup", s2, "Cup", s1
        os += -tdiag, "Cdagdn", s1, "Cdn", s2
        os += -tdiag, "Cdagdn", s2, "Cdn", s1
    end

    #Hubbard U term
    for i=1:N
        os += U, "Nupdn", i
    end

    #Extended NN interaction
    for (s1,s2) in bonds_rung
        os += -V, "Ntot", s1, "Ntot", s2
    end
    for (s1,s2) in vcat(bonds_l1, bonds_l2)
        os += -V, "Ntot", s1, "Ntot", s2
    end

    sites = siteinds("Electron", 2*L, conserve_qns=true)
    H = MPO(os, sites)

    #initial state
    ntot = Nup+Ndn
    select_sites = randperm(N)[1:ntot]
    state = ["Emp" for n in 1:N]
    for (id, s) in enumerate(select_sites)
            state[s] = isodd(id) ? "Up" : "Dn"
    end

    psi0 = productMPS(sites, state)
    Sz = expect(psi0,"Sz")
    println("Initial state Sz : ", Sz)
    println()

    nsweeps = 30
    maxdim = [10,20,40,80,200,200,200,200,500,500,500,500,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,3000,3000,3000,3000,5400,5400,5400,5400]
    cutoff = [1E-7]
    noise = [1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7, 1E-7, 1E-8, 1E-8, 1E-8,0.0,0.0,0.0]

    #energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
    obs = DMRGObserver(; energy_tol=1e-10)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff,noise,observer=obs,eigsolve_krylovdim=8)
    save_mps("psi_L$(L)_Nup$(Nup)_Ndn$(Ndn)_Vn$(V).h5",psi)

    println()
    @show expect(psi,"Sz")

    ninj = @time correlation_matrix(psi, "Ntot", "Ntot")
    didj = @time correlation_matrix(psi, "Nupdn", "Nupdn")CdagC = two_pt_correlator(psi)
    SziSzj = @time correlation_matrix(psi, "Sz", "Sz")
    ntot = expect(psi,"Ntot")
    dtot = expect(psi,"Nupdn")
    println()
    @show ntot
  
    fname = "data_L$(L)_Nup$(Nup)_Ndn$(Ndn)_Vn$(V).h5"
    h5write(fname, "ntot", ntot)
    h5write(fname, "dtot", dtot)
    h5write(fname, "ninj", ninj[:,L])
    h5write(fname, "didj", didj[:,L])
    h5write(fname, "SziSzj", SziSzj[:,L])

    CdagC = two_pt_correlator(psi)
    #CdagC = @time correlation_matrix(psi, "Cdagup", "Cup")
    @show CdagC
    h5write(fname, "CdagC", CdagC)
    return
end

    println()

    
