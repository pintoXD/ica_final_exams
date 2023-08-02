% Implementacao do algoritmo Local Random Search (LSR)
% para encontrar o minimo da funcao Ackley 1D
%
%       f(x) = -20exp(-0.2|x|)-exp(cos(2*pi*x))+20+exp(1),  -5 < x < 5.
%
% Autor: Guilherme A. Barreto
% Data: 02/12/2017

clear; clc; close all;

%%% Gráfico da função de Ackley
limites=[-5 5];   % Limites do intervalo de busca para funcao de Ackely
x=limites(1):0.1:limites(2);  % Dominio de x discretizado em incrementos de 0.1
f_ackley=ackley1D(x);
figure; plot(x,f_ackley,'k-');
hold on

%%% Parametros do LSR
Ng=500;   % Numero de iteracoes

dp=0.5;    % numero de particulas

x_best=unifrnd(limites(1),limites(2)); 	% Gera vetor de solucoes iniciais

Fbest=ackley1D(x_best);  % Avalia solucao inicial

%%% Roda algoritmo por Ng iteracoes
for t=1:Ng,
    iteracao=t;
    
    %%% Plota melhor solucao ate o momento sobre a curva da funcao de ackley
    plot(x_best,Fbest,'ro');

    ruido=normrnd(0,dp);     % Perturbacao aleatorio na melhor solucao

    x_cand = x_best + ruido; 	% Gera solucao candidata

    % Verifica se solucao candidata estah fora do limite superior
    if x_cand > limites(2),
	    x_cand = limites(2);
    end

    % Verifica se solucao candidata estah fora do limite inferior
    if x_cand < limites(1),
	    x_cand = limites(1);
    end

    Fcand=ackley1D(x_cand);   % Avalia solucao candidata
    
    if Fcand<Fbest,  % Se minimizacao: (Fcand<Fbest). Se maximizacao: (Fcand>Fbest)
        x_best=x_cand;    % Se x_cand produz melhor resultado que x_best, x_cand vira "melhor solucao ate o momento"
        Fbest=Fcand;
    end

    aptidao(t)=Fbest;

end

x_best, Fbest

%%% Plota melhor solucao apos Ng iteracoes sobre a curva da funcao de ackley
plot(x_best,Fbest,'ro');
hold off

figure; plot(aptidao);
xlabel('Iteration');
ylabel('Fitness');
