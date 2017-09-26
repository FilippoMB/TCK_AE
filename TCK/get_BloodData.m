function [ X, Y, Xte, Yte ] = get_BloodData(data_norm)
%GET_BLOODDATA Summary of this function goes here
%   Detailed explanation goes here

Xall = load('..\Data\X_all.mat');
Xall = Xall.X;
Yall = load('..\Data\Y_all.mat');
Yall = Yall.Y;

[N,T,V] = size(Xall);

if data_norm
    for v=1:V
       Xv = Xall(:,:,v);
       Xv_m = nanmean(Xv(:));
       Xv_s = nanstd(Xv(:));
       Xv = (Xv - Xv_m)/Xv_s;
       Xall(:,:,v) = Xv;
    end
end

X = Xall(1:719,:,:);
Xte = Xall(720:end,:,:);
Y = Yall(1:719,:,:);
Yte = Yall(720:end,:,:);
end

