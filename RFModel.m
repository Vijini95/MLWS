function [predicted_labels, model] = RFModel(XTrain, XTest,isDisplay)
% RFModel - Random Forest classification with robust imbalance handling

    %% 1. Separate features and labels
    features_train = XTrain(:, 1:end-1);
    labels_train = XTrain(:, end);
    
    %% 2. Check class distribution
    class_counts = histcounts(labels_train);
    if isDisplay == 1
    fprintf('Class distribution: %d (class 0) vs %d (class 1)\n', class_counts(1), class_counts(2));
    end
    %% 3. Alternative imbalance handling approaches
    
    % Option 1: Use Bagging instead of RUSBoost for extreme imbalance
    if min(class_counts) < 10 || (max(class_counts)/min(class_counts)) > 20
        fprintf('Using Bagging due to extreme imbalance\n');
        t = templateTree('MaxNumSplits', 20, 'MinLeafSize', 10);
        model = fitensemble(features_train, labels_train, 'Bag', 150, t, ...
                          'Type', 'Classification', 'ClassNames', [0; 1]);
    else
        % Option 2: Adjusted RUSBoost with careful parameters
        t = templateTree('MaxNumSplits', 10, ...  % Reduced splits
                        'MinLeafSize', 30, ...   % Larger leaf size
                        'NumVariablesToSample', ceil(size(features_train,2)/2));
        
        % Try RUSBoost first, fall back to Bagging if warning occurs
        try
            model = fitensemble(features_train, labels_train, 'RUSBoost', 100, t, ...
                              'LearnRate', 0.05, ...  % Lower learning rate
                              'ClassNames', [0; 1]);%, ...
                              %'Print', 1);
        catch ME
            warning('RUSBoost failed, falling back to Bagging: %s', ME.message);
            model = fitensemble(features_train, labels_train, 'Bag', 150, t, ...
                              'Type', 'Classification', 'ClassNames', [0; 1]);
        end
    end
    
    %% 4. Evaluate model with stratified cross-validation
    %rng(42); % For reproducibility
    cvp = cvpartition(labels_train, 'KFold', 5, 'Stratify', true);
    cvmodel = crossval(model, 'CVPartition', cvp);
    cvloss = kfoldLoss(cvmodel, 'LossFun', 'ClassifError');
    
    %% 5. Calculate comprehensive metrics
    [cv_labels, scores] = kfoldPredict(cvmodel);
    [X,Y,T,AUC] = perfcurve(labels_train, scores(:,2), 1);
    
    % Confusion matrix with adjusted decision threshold
    optimal_idx = find(X >= 0.7, 1); % Targeting at least 70% recall
    if isempty(optimal_idx)
        optimal_idx = length(X);
    end
    adjusted_threshold = T(optimal_idx);
    adjusted_labels = scores(:,2) >= adjusted_threshold;
    
    if isDisplay == 1
    C = confusionmat(labels_train, adjusted_labels);
    TP = C(2,2); FP = C(1,2); FN = C(2,1); TN = C(1,1);
    
    performance_metrics = struct(...
        'accuracy', (TP+TN)/sum(C(:)), ...
        'precision', TP/(TP+FP), ...
        'recall', TP/(TP+FN), ...
        'f1_score', 2*TP/(2*TP+FP+FN), ...
        'auc', AUC, ...
        'confusion_matrix', C, ...
        'optimal_threshold', adjusted_threshold);
    end
    %% 6. Predict on test data using optimal threshold
    [~, test_scores] = predict(model, XTest);
    predicted_labels = test_scores(:,2) >= adjusted_threshold;
    
    %% 7. Feature importance (if using Bagging or successful RUSBoost)
    if ~isempty(strfind(model.Method, 'Bag')) || ~isempty(strfind(model.Method, 'RUSBoost'))
        imp = predictorImportance(model);
        
         if isDisplay == 1
        figure;
        bar(imp);
        title('Feature Importance');
        ylabel('Importance Score');
        xlabel('Features');
        set(gca, 'XTickLabel', 1:size(features_train,2));
         end
    end
end