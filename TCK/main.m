% load data
%[ X, Y, Xte, Yte ] = get_BloodData(1,0.8);
[ X, Y, Xte, Yte ] = get_BloodData2(1);

%% Train GMM models
[GMMpar,C,G]  = trainTCK(X);

% Compute in-sample kernel matrix
Ktrtr = TCK(GMMpar,C,G,'tr-tr');

% Compute similarity between Xte and the training elements
Ktrte = TCK(GMMpar,C,G,'tr-te',Xte);

% Compute kernel matrix between test elements
Ktete = TCK(GMMpar,C,G,'te-te',Xte);

%% kNN -classifier
[acc, Ypred] = myKNN(Ktrte,Y,Yte,1);
[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean] = confusion_stats(Yte,Ypred);
disp(['acc: ',num2str(acc),', f1: ',num2str(f_measure)])

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

[~,idx] = sort(Yte);
Ksort = Ktete(idx,idx);
figure
imagesc(Ksort)
colormap('gray')
set(gca,'xtick',[])
set(gca,'ytick',[])
title('TCK K')

%% save mat files
save('TCK_data.mat', 'X','Y','Xte','Yte','Ktrtr','Ktrte','Ktete')
