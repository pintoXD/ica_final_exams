function F=pso_sea(b_aerogerador, v_aero, P)
  SEA=[];
  Ng= size(b_aerogerador)(1);
  for i=1:Ng,
    ypred=polyval(b_aerogerador(i, :), v_aero);
    y = P;
    erro = y - ypred;
    SEA = [SEA; sum(abs(erro))];
  end
F=SEA;  % Avalia solucoes iniciais
