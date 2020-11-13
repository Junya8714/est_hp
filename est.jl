using LinearAlgebra
using DelimitedFiles
using Distributions
using Plots
using Random

mutable struct Agents_Struct
    dx::Array{Float64,2}
    dy::Array{Float64,2}
    theta::Array{Float64,2}
    y::Array{Float64,2}
    z::Array{Float64,2}
    pi_til::Array{Float64,2}
    y_R::Array{Float64,3}
    z_R::Array{Float64,3}
    grad_L::Array{Float64,2}
end
function Grad_L(theta, dx, dy)#各エージェントのカーネル計算から勾配
    grad_L = zeros(Float64,10,3)
    grad_K3 = Matrix(1.0I, 100, 100)
    for i in 1:10
        x_diff = dx[i,:] * ones(1, 100) - (dx[i,:] * ones(1, 100))'#dxの差（100×100の対象行列）
        K = theta[i,1]  .* exp.((-(x_diff) .^ 2) ./ theta[i,2]) + Matrix(theta[i,3]I, 100, 100)
        A = -(x_diff) .^ 2 ./ theta[i,2]
        K_Inv = inv(K)
        grad_K1 = exp.(A)
        grad_K2 = theta[i,1] .* grad_K1 .* ((x_diff) ./ theta[i,2]) .^ 2
        M = K_Inv * dy[i,:]
        grad_L[i,1] = tr(K_Inv * grad_K1) - M' * grad_K1 * M
        grad_L[i,2] = tr(K_Inv * grad_K2) - M' * grad_K2 * M
        grad_L[i,3] = tr(K_Inv * grad_K3) - M' * grad_K3 * M
    end
    return grad_L
end

function Step1(theta, grad_L, pi_til, k)
    alpha::Float64 = 1/(k+1000)
    tau::Int64 = 20
    z=zeros(Float64,10,3)
    for i in 1:10
        z[i,:] = theta[i,:]  + alpha .* (clamp.(theta[i,:] - (grad_L[i,:] + pi_til[i,:]) ./ tau, 1e-5, 100) - theta[i,:])
    end
    return z
end
function Step3(z, grad_L, pi_til, dx, dy, z_R, y_R,w, N=10)
    theta = similar(z)
    y = similar(z)
    pi_til = similar(z)
    new_grad_L = zeros(Float64,10,3)
    for i in 1:10
        theta[i,:] = z[i,:] - z_R[i,:,i] + z_R[i,:,:] * w[i,:]
        new_grad_L! = Grad_L(theta,dx,dy)
        y[i,:] = new_grad_L[i,:] - grad_L[i,:] + y_R[i,:,:] * w[i,:]
        pi_til[i,:] = 10.0 .* y[i,:] - new_grad_L[i,:]
    end
    theta, new_grad_L, y, pi_til
end
function ite(Agents)
    iteration::Int64 = 10
    send_number::Int64 = 0
    weight = [
        0.4 0.2 0.2 0.2 0 0 0 0 0 0
        0.2 0.4 0.2 0 0.2 0 0 0 0 0
        0.2 0.2 0.2 0.2 0.2 0 0 0 0 0
        0.2 0 0.2 0.4 0 0.2 0 0 0 0
        0 0.2 0.2 0 0.4 0 0.2 0 0 0
        0 0 0 0.2 0 0.4 0 0.2 0.2 0
        0 0 0 0 0.2 0 0.4 0.2 0 0.2
        0 0 0 0 0 0.2 0.2 0.2 0.2 0.2
        0 0 0 0 0 0.2 0 0.2 0.4 0.2
        0 0 0 0 0 0 0.2 0.2 0.2 0.4
    ]
    for k in 1:iteration
        #E_z = 0
        #E_y = 0
        """step1:代替関数の最小化"""
        Agents.z = Step1(Agents.theta, Agents.grad_L, Agents.pi_til, k )
        """Step2:情報交換"""
        for i = 1:10
            #if norm(Agents.z-Agents.z_R::Real=2)>=E_z || norm(Agents.y-Agents.y_R::Real=2)>=E_y || k==0
            for j = 1:10
                if weight[j, i] != 0
                    Agents.z_R[j,:, i] = Agents.z[i,:]
                    Agents.y_R[j,:, i] = Agents.y[i,:]
                    if i != j
                        send_number += 1
                    end
                end
            end
        end
        """Step3:更新"""
        Agents.theta, Agents.grad_L, Agents.y, Agents.pi_til = Step3(
            Agents.z,
            Agents.grad_L,
            Agents.pi_til,
            Agents.dx,
            Agents.dy,
            Agents.z_R,
            Agents.y_R,
            weight
            )
    end
end


"""main関数"""
function Est()
    init_theta = [
        1.0 1.0 0.01
        0.6 0.1 0.013
        0.6 1.3 0.005
        1.5 0.4 0.012
        0.74 0.8 0.006
        2.0 0.5 0.009
        1.2 0.71 0.015
        1.21 0.6 0.011
        2.0 1.0 0.008
        0.7 0.9 0.0095
    ]
    Random.seed!(1)
    D_x = randn(Float64, (10, 100))
    D_y = sin.(pi .* D_x) + rand(Normal(0, 0.1), Base.size(D_x))
    Agents = Agents_Struct(
            D_x,
            D_y,
            init_theta,
            zeros(Float64, 10,3),
            zeros(Float64, 10,3),
            zeros(Float64, 10,3),
            zeros(Float64, 10,3 ,10),
            zeros(Float64, 10,3 ,10),
            zeros(Float64, 10,3),
        )
    Agents.grad_L = Grad_L(Agents.theta, Agents.dx, Agents.dy)
    Agents.y, Agents.pi_til = Agents.grad_L, 9.0 .* Agents.grad_L
#反復
    @time ite(Agents)
    println(mean(Agents.theta,dims=1))
end
Est()
