function F=pso_seq(b_aerogerador, v_aero, P)
  SEQ=[];
  Ng= size(b_aerogerador)(1);
  for i=1:Ng,
    ypred=polyval(b_aerogerador(i, :), v_aero);
    y = P;
    erro = y - ypred;
    SEQ = [SEQ; sum(erro.^2)];
  end
F=SEQ;  % Avalia solucoes iniciais
