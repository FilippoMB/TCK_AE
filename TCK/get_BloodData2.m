function [ X, Y, Xte, Yte ] = get_BloodData2(data_norm)

x_tr = load('../Data/x.mat');
y_tr = load('../Data/Y.mat');
x_te = load('../Data/xte.mat');
y_te = load('../Data/Yte.mat');

Ntr = size(x_tr.x,1);
Nts = size(x_te.xte,1);
T = 20;
V = 10;

X = reshape(x_tr.x,[Ntr,T,V]);
Xte = reshape(x_te.xte,[Nts,T,V]);
Y = y_tr.Y;
Yte = y_te.Yte;

if data_norm
    for v=1:V
       X_v = X(:,:,v);
       Xte_v = Xte(:,:,v);
       Xv_m = nanmean(X_v(:));
       Xv_s = nanstd(X_v(:));
       
       X_v = (X_v - Xv_m)/Xv_s;
       X(:,:,v) = X_v;
       Xte_v = (Xte_v - Xv_m)/Xv_s;
       Xte(:,:,v) = Xte_v;
    end
end

end