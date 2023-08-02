% Implementacao do algoritmo de busca aleatoria GLOBAL (GRS, global random search)
% para encontrar o o minimo da funcao Ackley 1D
%
%       f(x) = -20exp(-0.2|x|)-exp(cos(2*pi*x))+20+exp(1),  -5 < x < 5.
%
% Autor: Guilherme A. Barreto
% Data: 29/11/2017

clear; clc; close all;
pkg load statistics

%%% Gráfico da função de Ackley
limites=[-5 5];   % Limites do intervalo de busca para funcao de Ackely
x=limites(1):0.1:limites(2);  % Dominio de x discretizado em incrementos de 0.1
f_ackley=ackley1D(x);
figure; plot(x,f_ackley,'k-');
hold on

%%% Parametros do AG
Ng=500;   % Numero de iteracoes

%%% Parametros do AG
Ng=1000;   % Numero de iteracoes

x_best=unifrnd(limites(1),limites(2)); 		% Gera solucao inicial dentro do intervalo e a faz a melhor ateh o momento.

Fbest=ackley1D(x_best);  % Avalia solucao inicial

%%% Roda AG por Ng geracoes
for t=1:Ng,
    iteracao=t;

    %%% Plota melhor solucao ate o momento sobre a curva da funcao de ackley
    plot(x_best,Fbest,'ro');

    x_cand=unifrnd(limites(1),limites(2)); 		% Gera solucao candidata

    Fcand=ackley1D(x_cand);  % Avalia solucao candidata

    if Fcand<Fbest, % Se minimizacao: (Fcand<Fbest). Se maximizacao: (Fcand>Fbest)
        x_best=x_cand;    % Se x_cand produz melhor resultado que x_best, x_cand vira "melhor solucao ate o momento"
        Fbest=Fcand;
    end

    aptidao(t)=Fbest;

end

x_best, Fbest

%%% Plota melhores solucoes apos Ng iteracoes sobre a curva da funcao de ackley
plot(x_best,Fbest,'ro');
hold off

figure; plot(aptidao);
xlabel('Iteration');
ylabel('Fitness');
