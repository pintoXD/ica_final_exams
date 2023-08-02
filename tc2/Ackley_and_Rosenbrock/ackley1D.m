function F=ackley1D(x)
% 
% Funcao objetivo Ackley 1D (1 variavel)
%
% Entradas:
%	x: valor a ser avaliado
%       funcao: f(x) = -20*exp(-0.2*abs(x))-exp(cos(2*pi*x))+20+exp(1);
%
% Saida:
%	F: valor da funcao-objetivo em x
%
% Data: 29/11/2017
% Autor: Guilherme Barreto

F=-20*exp(-0.2*abs(x))-exp(cos(2*pi*x))+20+exp(1);
