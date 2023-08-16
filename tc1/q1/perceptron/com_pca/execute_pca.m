function x_with_pca = execute_pca(Xc)
    # pkg load statistics
    Cx = cov(Xc');  % Estima a matriz de covariancia dos dados simulados
    [V L]=eig(Cx); L=diag(L);
    [L I]=sort(L,'descend'); V=V(:,I);
    VEi=100*L/sum(L);  % Variancia explicada pelo i-esimo autovalor
    VEq=100*cumsum(L)/sum(L);   % Variancia explicada pelos q primeiros autovalores

    % figure; plot(VEq,'r-','linewidth',2);
    % xlabel('Numero de autovalores principais (q)'); ylabel('Variancia Explicada');
    % set(gca, "fontsize", 14)

    [V2 L2 VEi2] = pcacov(Cx);  % Calcula autovalores e autovetores da matriz Cx
    VEq2=cumsum(VEi);

    [U3 L3 V3]=svd(Cx); %%%% PCA a partir da SVD %%%%%%

    Q = V';  % Monta matriz de transformacao (sem reducao de dimensionalidade)
    % Q=V2';
    % Q=V3';

    Z = Q*Xc;  % Gera dados via PCA (descorrelaciona matriz dos dados)
  
    x_with_pca = Z;

endfunction