% load data
[ X, Y, Xte, Yte ] = get_BloodData(0);

%% Train GMM models
[GMMpar,C,G]  = trainTCK(X);

% Compute in-sample kernel matrix
Ktrtr = TCK(GMMpar,C,G,'tr-tr');

% Compute similarity between Xte and the training elements
Ktrte = TCK(GMMpar,C,G,'tr-te',Xte);

% Compute kernel matrix between test elements
Ktete = TCK(GMMpar,C,G,'te-te',Xte);

%% kNN -classifier
accuracy=myKNN(Ktrte,Y,Yte,9)

%% visualization
close all
[~, score] = pca(Ktete);
X_proj = score(:,1:2);
figure
hold on
for i=1:max(Yte)
    plot(score(Yte==i,1),score(Yte==i,2),'.','markersize',10)
end
set(gca,'xtick',[])
set(gca,'ytick',[])
box on
title('PCA TCK')

figure
imagesc(Ktete)
colormap('gray')
set(gca,'xtick',[])
set(gca,'ytick',[])
title('TCK K')

%% save mat files
save('TCK_data.mat', 'X','Y','Xte','Yte','Ktrtr','Ktrte','Ktete')