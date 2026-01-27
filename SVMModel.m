function [predicted_labels, model, performance_metrics] = SVMModel(XTrain, XTest, isDisplay)
% SVMModel - Performs binary SVM classification with class imbalance handling
%
% Inputs:
%   XTrain - Training data matrix where last column contains labels (0 or 1)
%   XTest  - Test data matrix (unlabeled) for prediction
%
% Outputs:
%   predicted_labels    - Predicted labels for XTest
%   model              - Trained SVM model
%   performance_metrics - Structure containing evaluation metrics

    %% 1. Separate features and labels from training data
    features_train = XTrain(:, 1:end-1);
    labels_train = XTrain(:, end);
    
    %% 2. Check class distribution  
    if isDisplay == 1
    class_counts = histcounts(labels_train);
    fprintf('Class distribution: %d (class 0) vs %d (class 1)\n', ...
           class_counts(1), class_counts(2));
    end
    %% 3. Standardize features (important for SVM)
    [features_train_scaled, mu, sigma] = zscore(features_train);
    features_test_scaled = (XTest - mu) ./ sigma;
    
    %% 4. Handle class imbalance using class weighting
    % Calculate class weights inversely proportional to class frequencies
    weight_positive = sum(labels_train == 0)/length(labels_train);
    weight_negative = sum(labels_train == 1)/length(labels_train);
    
    % Create SVM template with weights
    t = templateSVM('BoxConstraint', 1, ...
                    'KernelFunction', 'rbf', ...
                    'Standardize', false, ... % already standardized
                    'ClassNames', [0; 1], ...
                    'Cost', [0, weight_negative; weight_positive, 0]);
    
    %% 5. Train the SVM model
    model = fitcecoc(features_train_scaled, labels_train, ...
                    'Learners', t, ...
                    'Coding', 'onevsone', ...
                    'Verbose', 0);
    
    %% 6. Evaluate model using cross-validation
    cvmodel = crossval(model, 'KFold', 10);
    cvloss = kfoldLoss(cvmodel, 'LossFun', 'ClassifError');
    
    % Get predicted labels from cross-validation
    [cv_labels, ~] = kfoldPredict(cvmodel);
    
    % Calculate performance metrics
    C = confusionmat(labels_train, cv_labels);
    TP = C(2,2); FP = C(1,2); FN = C(2,1); TN = C(1,1);
    
    performance_metrics = struct();
    performance_metrics.accuracy = 1 - cvloss;
    performance_metrics.precision = TP/(TP+FP);
    performance_metrics.recall = TP/(TP+FN);
    performance_metrics.f1_score = 2*(performance_metrics.precision*performance_metrics.recall)/...
                                  (performance_metrics.precision+performance_metrics.recall);
    performance_metrics.confusion_matrix = C;
    
    if isDisplay == 1
        % Display performance metrics
        fprintf('\nPerformance Metrics:\n');
        fprintf('Accuracy: %.2f%%\n', performance_metrics.accuracy*100);
        fprintf('Precision: %.2f\n', performance_metrics.precision);
        fprintf('Recall: %.2f\n', performance_metrics.recall);
        fprintf('F1 Score: %.2f\n', performance_metrics.f1_score);
        disp('Confusion Matrix:');
        disp(performance_metrics.confusion_matrix);
    end
    
    %% 7. Predict labels for test data
    predicted_labels = predict(model, features_test_scaled);
    
    %% 8. Optional: Get decision scores or probabilities
    % Uncomment if you need probability estimates
    % score_model = fitSVMPosterior(model);
    % [~, posterior_prob] = predict(score_model, features_test_scaled);
end