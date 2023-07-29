function F=func_objetivo2D(x,y,funcao)
% 
% Funcao objetivo 2D (2 variaveis: x e y)
%
% Entradas:
%	x: valor da variavel 1
%	y: valor da variavel 2
%
%       funcao: 1, se f(x,y)=1/((x-20)^2 + (y-20)^2 + 50)
%		2, se f(x,y))=60x + 100y -1.5x - 1.5y - xy
%
% Saida:
%	F: valor da funcao-objetivo em (x,y)
%
% Data: 22/11/2017
% Autor: Guilherme Barreto

if funcao==1,
	den=(x-20)^2 + (y-20)^2 + 50;
	F=1/den;
else   
	F=60*x + 100*y -1.5*x*x - 1.5*y*y - x*y;
end
