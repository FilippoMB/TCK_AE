% load data
[ X, Y, Xte, Yte ] = get_BloodData(1);

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
[~,~,~,AUC] = perfcurve(Yte,Ypred,1);
disp(['ACC: ',num2str(acc),', F1: ',num2str(f_measure),', AUC: ',num2str(AUC)])

%% visualization

[~,idx] = sort(Yte);
Ksort = Ktete(idx,idx);
figure
imagesc(Ksort)
colormap('gray')
set(gca,'xtick',[])
set(gca,'ytick',[])
title('TCK K')

%% save mat files
save('../Data/TCK_data.mat', 'X','Y','Xte','Yte','Ktrtr','Ktrte','Ktete')
