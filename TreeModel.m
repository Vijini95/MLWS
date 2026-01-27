function [predicted_labels, model] = TreeModel(XTrain, XTest, isDisplay)
% TreeModel - Decision Tree classification with hyperparameter optimization
% and class imbalance handling
%
% Inputs:
%   XTrain - Training data matrix where last column contains labels (0 or 1)
%   XTest  - Test data matrix (unlabeled) for prediction
%
% Outputs:
%   predicted_labels - Predicted labels for XTest
%   model - Trained Decision Tree model

    %% 1. Separate features and labels
    features_train = XTrain(:, 1:end-1);
    labels_train   = XTrain(:, end);

    %% 2. Check class distribution
    class_counts = histcounts(labels_train);
    imbalance_ratio = max(class_counts)/min(class_counts);
    if isDisplay == 1
        fprintf('Class distribution: %d (class 0) vs %d (class 1)\n', class_counts(1), class_counts(2));
        fprintf('Imbalance ratio: %.2f:1\n', imbalance_ratio);
    end

    %% 3. Set up class weights for imbalance
    class_weights = 1./class_counts;
    cost_matrix = [0, class_weights(2); class_weights(1), 0];

    %% 4. Hyperparameter Optimization Setup
    cvp = cvpartition(labels_train, 'KFold', 5, 'Stratify', true);

    vars = [
        optimizableVariable('MaxNumSplits', [1, 100], 'Type', 'integer')
        optimizableVariable('MinLeafSize', [1, 50],  'Type', 'integer')
        optimizableVariable('SplitCriterion', {'gdi','twoing','deviance'}, 'Type', 'categorical')
    ];

    %% 5. Run Bayesian Optimization
    optim_results = bayesopt(@(params)tree_objective(params, features_train, labels_train, cost_matrix, cvp), ...
                           vars, ...
                           'MaxObjectiveEvaluations', 30, ...
                           'UseParallel', false, ...
                           'PlotFcn', [], ...
                           'AcquisitionFunctionName', 'expected-improvement-plus', ...
                           'Verbose', 0);

    %% 6. Extract best parameters (handle struct vs table)
    best_params = optim_results.XAtMinObjective;
    if istable(best_params)
        best_params = table2struct(best_params);
    end

    %% 7. Train final model
    model = fitctree(features_train, labels_train, ...
        'MaxNumSplits', best_params.MaxNumSplits, ...
        'MinLeafSize',  best_params.MinLeafSize, ...
        'SplitCriterion', char(best_params.SplitCriterion), ...
        'Cost', cost_matrix, ...
        'Surrogate', 'on', ...
        'Prune', 'on', ...
        'ClassNames', [0; 1]);

    %% 8. Cross-validation (optional)
    cvmodel = crossval(model, 'CVPartition', cvp);
    [~, scores] = kfoldPredict(cvmodel);

    %% 9. ROC & optimal threshold
    [X,Y,T,~] = perfcurve(labels_train, scores(:,2), 1);
    [~, optimal_idx] = max(2*Y.*X./(Y+X+eps));
    optimal_threshold = T(optimal_idx);

    %% 10. Predict on test data
    [~, test_scores] = predict(model, XTest);
    predicted_labels = test_scores(:,2) >= optimal_threshold;
end

%% Objective function for Bayesian Optimization
function objective = tree_objective(params, X, y, cost_matrix, cvpartition)
    temp_model = fitctree(X, y, ...
        'MaxNumSplits', params.MaxNumSplits, ...
        'MinLeafSize',  params.MinLeafSize, ...
        'SplitCriterion', char(params.SplitCriterion), ...
        'Cost', cost_matrix, ...
        'CVPartition', cvpartition);

    pred = kfoldPredict(temp_model);
    C = confusionmat(y, pred);
    TP = C(2,2); FP = C(1,2); FN = C(2,1);
    precision = TP/(TP+FP+eps);
    recall    = TP/(TP+FN+eps);
    f1        = 2*(precision*recall)/(precision+recall+eps);

    objective = -f1; % Minimize negative F1 score
end
