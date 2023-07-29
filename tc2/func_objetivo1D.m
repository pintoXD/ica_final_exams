function F=func_objetivo1D(x,funcao)
% 
% Funcao objetivo 1D (1 variavel)
%
% Entradas:
%	x: valor a ser avaliado
%       funcao: 1, se f(x)=x^2
%		2, se f(x)=x*sin(10*pi*x)+1;
%
% Saida:
%	F: valor da funcao-objetivo em x
%
% Data: 22/11/2017
% Autor: Guilherme Barreto

if funcao==1,
       F=x.*x;
else   F=x.*sin(10*pi*x)+1;
end
