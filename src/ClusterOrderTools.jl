module ClusterOrderTools

using Clustering, Random, Statistics
using ProgressMeter

export kmeansorder, evalclustering, average_heatmap, average_heatmap_vert

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
    
    
    aic = ci + 2*size(X, 1).*ks

    (ks=ks, randindex=ri, varinfo=vi, costs=ci, rtot =rtot, rsquared=r², bic=bic, aic=aic)
end


function average_heatmap(H::Vector{T}, δ) where {T}
    n = length(H)

    w = cld(n, δ)
    AH = zeros(Float64, w)
    te = zeros(Int, w)
    for i = 1:n
        wi = cld(i, δ)
        te[wi] += 1
        AH[wi] += H[i]
    end
    if te[end] < 0.75δ
        AH[end] += AH[end-1]
        te[end] += te[end-1]
    end
    AH./te
end


average_heatmap(H, δh, δv) = average_heatmap_vert(average_heatmap(H, δh), δv)

function average_heatmap_vert(H, δ=2)
    
    n, m = size(H)
    
    w = cld(n, δ)
    AH = zeros(Float64, w, m)
    te = zeros(Int, w)
    for i = 1:m
        for j = 1:n
            wj = cld(j, δ)
            (i == 1) && (te[wj] += 1)
            AH[wj, i] += H[j, i]
        end
    end

    if te[end] < 0.75δ
        for i = 1:m
            AH[end, i] += AH[end-1, i]
        end
        te[end] += te[end-1]
    end

    AH./te
end

function average_heatmap(H, δ=2)
    
    n, m = size(H)
    w = cld(m, δ)
    
    AH = zeros(Float64, n, w)
    te = zeros(Int, w)
    for i = 1:m
        wi = cld(i, δ)
        te[wi] += 1
        for j = 1:n
            AH[j, wi] += H[j, i]
        end
    end
    if te[end] < 0.75δ
        for j = 1:n
            AH[j, end] += AH[j, end-1]
        end
        te[end] += te[end-1]
    end
    AH./te'
end


end # module
