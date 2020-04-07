module ClusterOrderTools

using Clustering, Random, Statistics
using Plots, ProgressMeter

export kmeansorder, evalclustering

function kmeansorder(X, k, sr=1618)
    if k == 1
       return (KSI=[1], A=ones(size(X, 2)), SI=1:size(X, 2), k=1) 
    end
    Random.seed!(sr)
    KM = kmeans(X, k)
    KSI = sortperm([findmax(@view KM.centers[:, i])[2] for i = 1:k], rev=false)
    KSSI = sortperm(KSI)
    A = KSSI[KM.assignments]
    SI = sortperm(A);
    
    (KSI=KSI, A=A, SI=SI, k=k, KM=KM, X=X)
end


function evalclustering(X, ks=2:2:40)
    KMK = @showprogress [kmeansorder(X, k) for k in ks]

    A = getfield.(KMK, :A)

    ri = [randindex(A[i], A[i+1]) for i = 1:(length(ks)-1)]
    vi = [varinfo(ks[i], A[i], ks[i+1], A[i+1]) for i = 1:(length(ks)-1)]
    ci = sum.(getfield.(getfield.(KMK, :KM), :costs))
    rtot = sum((X .- mean(X, dims=2)).^2)
    r² = 1 .- ci./rtot
    n, m = size(X)
    
    bic = n*log.(ci./(n*m)) + ks.*log(n)
    
    [ci ks 2*size(X, 1).*ks] |> display
    aic = ci + 2*size(X, 1).*ks
    phs = [plot(ks[1:end-1], getindex.(ri, i), xticks=ks, marker=(stroke(:white))) for i = 1:4]

    push!(phs, plot(ks[1:end-1], vi, xticks=ks, marker=(stroke(:white))))
    push!(phs, plot(ks, r², xticks=ks, marker=(stroke(:white))))
    
    plot(phs..., layout=(1, length(phs)), size=(1500, 300), leg=false)
end



end # module
