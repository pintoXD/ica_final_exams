function x_with_pca = execute_pca(Xc)
    pkg load statistics
    internal_Xc = Xc;
    Z = 0;

    Cx = cov(Xc');  % Estima a matriz de covariancia dos dados simulados

    [COEFF LATENT EXPLAINED] = pcacov(Cx);  % Calcula autovalores e autovetores da matriz Cx
    variance_sum = 0;
    index_max_variance_sum = 0;

    for i=1:size(EXPLAINED)(1),
        variance_sum = variance_sum + EXPLAINED(i,:);
        index_max_variance_sum = i;
        if (variance_sum >= 90.0)
            break;
        endif
    end
    
    Q=COEFF(:,1:index_max_variance_sum);
    % Q=COEFF(1:index_max_variance_sum, :);
    Z = Q'*internal_Xc;  % Gera dados via PCA (descorrelaciona matriz dos dados)
    % Z = Q*internal_Xc;  % Gera dados via PCA (descorrelaciona matriz dos dados)
    size_Z = size(Z)
    x_with_pca=Z;
endfunction