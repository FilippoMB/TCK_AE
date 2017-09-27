function [ acc, labels_pred ] = myKNN( S, labels_tr, labels_ts, k )
%MYKNN  
% Input:
% S - similarity matrix
% labels - class labels
% k - num of neighbors

    labels_pred = zeros(size(labels_ts));
    for i=1:size(S,2)
        S_i = S(:,i);
        [~,c] = sort(S_i,'descend');
        c = c(1:k);
        L = labels_tr(c);
        labels_pred(i) = mode(L);

    end
    
    acc = sum(labels_pred == labels_ts)/length(labels_ts);    

end

