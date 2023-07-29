% Implementacao do algoritmo de busca aleatoria GLOBAL (GRS, global random search)
% para encontrar o maximo/minimo de funcoes de 1 variavel
%       f(x) = x^2,   0 < x < 31;
%       f(x) = x*sin(10*pi*x)+1,  -1 < x < 2.
%
% Autor: Guilherme A. Barreto
% Data: 29/11/2017

clear; clc; close all;

pkg load statistics
load aerogerador.dat
v=aerogerador(:,1);
P=aerogerador(:,2);

%%% Parametros do GRS
Ng=1000;   % Numero de iteracoes

% funcao=2;  % Escolhe funcao-objetivo

% Escolhe limites do dominio de x
% if funcao==1,
% 	limites=[0 31];   % Limites do intervalo para funcao: F=x^2
% else    limites=[-1 2];   % Limites do intervalo de busca para funcao: F=x*sin(10*pi*x)+1;
% end

%%%%%%%%%%% Grafico da funcao a ser otimizada
% x=limites(1):0.01:limites(2);  % Dominio de x discretizado em incrementos de 0.1
% f_opt=func_objetivo1D(x,funcao);  % Avalia solucao inicial
%figure; plot(x,f_opt,'k-');
%hold on
%%%%%%%%%%%

limites=[-6 11];
k = 4;
x_best=(unifrnd(limites(1), limites(2), 1, k+1)); 	% Gera solucao inicial dentro do intervalo permitido
% x_best=[-6.7670e-02   1.4339e+00  -5.0036e+00   1.0373e+01  -3.9133e-01]
% x_best=[-0.1521   1.6590  -0.4391  -4.2518  -3.2364]
% B=polyfit(v, P, k_best);
ypred = polyval(x_best, v);  % Avalia solucao candidata
y = P;
erro = y - ypred;
SEQ = sum(erro.^2);

Fbest=SEQ;
Fcand=Fbest
% Fbest=func_objetivo1D(x_best,funcao);  % Avalia solucao inicial

cands = [];
% t = 1;
%pause(2);
%%% Roda GRS por Ng geracoes
for t=1:Ng,
% while (Fcand > 1)
    % iteracao=t;

    x_cand=(unifrnd(limites(1), limites(2), 1, k+1));
    cands = [cands; x_cand];
    ypred = polyval(x_cand, v);
    erro = y - ypred;
    SEQ = sum(erro.^2); % Gera solucao candidata

    Fcand=SEQ;

    % Avalia solucao candidata
    if Fcand<Fbest,	% Se x_cand produz melhor resultado que x_best
        x_best=x_cand;  % x_cand vira "melhor solucao ate o momento"
        Fbest=Fcand;
    end

    % plot(x_best,Fbest,'ro');  %%% Plota melhor solucao (solucoes intermediarias)
    % t++;
    aptidao(t)=Fbest;
end
% endwhile

x_best, Fbest

%plot(x_best,Fbest,'b^');  %%% Plota melhor solucao
%hold off

figure; plot(aptidao,'linewidth',3);
xlabel('Iteracao'); ylabel('y=f(x)');
set(gca, "fontsize", 14);



figure; plot(v,ypred);  %%% Plota solucao final
title("GRS");
