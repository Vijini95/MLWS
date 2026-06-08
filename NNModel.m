function [predicted_labels, net] = NNModel(XTrain, XTest, isDisplay)
% NNModel - Neural Network classification with class imbalance handling (safe version)
%
% Inputs:
%   XTrain - Training data matrix where last column contains labels (0 or 1)
%   XTest  - Test data matrix (unlabeled) for prediction
%   isDisplay - (0/1) flag to print training info
%
% Outputs:
%   predicted_labels - Predicted labels for XTest (0 or 1)
%   net - Trained neural network

    %% 1. Separate features and labels
    features_train = XTrain(:, 1:end-1)';
    labels_train = XTrain(:, end)';

    %% 2. Convert labels to [1 0] / [0 1] format for patternnet
    target_matrix = full(ind2vec(labels_train + 1, 2));

    %% 3. Handle extreme imbalance
    if numel(unique(labels_train)) < 2
        warning('Only one class found in training labels. Adding synthetic minority sample.');
        labels_train = [labels_train, 1 - labels_train(1)]; % force 0 and 1
        features_train = [features_train, features_train(:,1)]; % duplicate feature vector
        target_matrix = full(ind2vec(labels_train + 1, 2));
    end

    class_counts = histcounts(labels_train, 0:2);
    imbalance_ratio = max(class_counts)/max(1,min(class_counts));
    if isDisplay == 1
        fprintf('Class distribution: %d (class 0) vs %d (class 1)\n', class_counts(1), class_counts(2));
        fprintf('Imbalance ratio: %.2f:1\n', imbalance_ratio);
    end

    %% 4. Class weighting for imbalance
    class_weights = 1./max(class_counts,1);
    sample_weights = ones(size(labels_train));
    sample_weights(labels_train == 0) = class_weights(1);
    sample_weights(labels_train == 1) = class_weights(2);

    %% 5. Neural network setup
    hiddenLayerSize = [10 5];
    net = patternnet(hiddenLayerSize);
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.3;
    net.trainFcn = 'trainscg';
    net.performFcn = 'crossentropy';
    net.trainParam.epochs = 1000;
    net.trainParam.max_fail = 100;

    %% 6. Train
    [net, tr] = train(net, features_train, target_matrix, {}, {}, {}, sample_weights);

    %% 7. Evaluate
    train_scores = net(features_train);
    [~, train_pred_labels] = max(train_scores);
    train_pred_labels = train_pred_labels - 1;

    val_indices = tr.valInd;
    val_scores = train_scores(2, val_indices);
    val_labels = labels_train(val_indices);

    %% === Safe ROC computation ===
    if numel(unique(val_labels)) < 2
        warning('Skipping perfcurve: validation labels have only one class. Setting AUC=NaN.');
        AUC = NaN; optimal_threshold = 0.5;  % fallback
    else
        [X,Y,T,AUC] = perfcurve(val_labels, val_scores, 1);
        [~, optimal_idx] = max(2*Y.*X./(Y+X+eps));
        optimal_threshold = T(optimal_idx);
    end

    %% 8. Predict on test data
    test_scores = net(XTest');
    predicted_labels = double(test_scores(2,:) > optimal_threshold)';

end
