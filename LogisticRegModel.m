function test_pred_class = LogisticRegModel(Xtrain, Xtest, isdisp)
        
    % Data Preparation
    X = Xtrain; % 1019x4 matrix (last column is binary response)
    X_test = Xtest; % Test data without response
    
    % 1. Split data and remove response
    predictors = X(:,1:end-1);
    response = X(:,end);

    class_counts = histcounts(response);
    
    % 2. Check for rank deficiency
    rankX = rank(predictors);
    [num_samples, num_features] = size(predictors);
    %disp(['Matrix rank: ', num2str(rankX), ' out of ', num2str(num_features), ' features']);
    
    % 3. Option 1: Remove redundant features automatically
    [Q, R, perm] = qr(predictors, 0);
    tol = max(size(predictors)) * eps(norm(R));
    rank_def = sum(abs(diag(R)) > tol);
    
    if rank_def < num_features
        disp(['Removing ', num2str(num_features-rank_def), ' redundant features']);
        predictors = predictors(:,perm(1:rank_def));
    end
    
    % 4. Split into training and validation
    %rng(42); % For reproducibility
   cv = cvpartition(response, 'HoldOut', 0.3, 'Stratify', true);
    X_train = predictors(training(cv),:);
    y_train = response(training(cv));
    X_val = predictors(test(cv),:);
    y_val = response(test(cv));
    
    % 3. Compute class weights (inverse frequency)
    class_weights = 1 ./ class_counts;
    class_weights = class_weights / sum(class_weights); % Normalize
    sample_weights = ones(size(y_train));
    sample_weights(y_train == 0) = class_weights(1);
    sample_weights(y_train == 1) = class_weights(2);
    
    % 4. Train logistic regression with class weights and regularization
    model = fitclinear(X_train, y_train, ...
        'Learner', 'logistic', ...
        'Regularization', 'ridge', ...
        'Lambda', 'auto', ...
        'Weights', sample_weights, ... % Class weighting
        'Solver', 'lbfgs', ...
        'ClassNames', [0, 1], ...
        'ScoreTransform', 'logit');
    
    % 5. Alternative: Use RUSBoost for severe imbalance
    % ensemble = fitcensemble(X_train, y_train, ...
    %     'Method', 'RUSBoost', ...
    %     'Learners', 'tree', ...
    %     'RatioToSmallest', [1 1], ... % Balance classes
    %     'NumLearningCycles', 100);
    
    % 6. Evaluate using balanced metrics
    val_pred_prob = predict(model, X_val);
    val_pred_class = double(val_pred_prob >= 0.5);
    
    % Standard metrics
    confusion_mat_val = confusionmat(y_val, val_pred_class);
    accuracy_val = sum(diag(confusion_mat_val))/sum(confusion_mat_val(:));
    
    % Balanced metrics
    TP = confusion_mat_val(2,2);
    TN = confusion_mat_val(1,1);
    FP = confusion_mat_val(1,2);
    FN = confusion_mat_val(2,1);
    
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    specificity = TN / (TN + FP);
    f1_score = 2 * (precision * recall) / (precision + recall);
    balanced_accuracy = (recall + specificity) / 2;
    
    % ROC analysis
    [fpr, tpr, ~, auc] = perfcurve(y_val, val_pred_prob, 1);
    
    % Display comprehensive results
    if isdisp == 1
        disp('=== Imbalanced Classification Results ===');
        disp(['Standard Accuracy: ', num2str(accuracy_val*100), '%']);
        disp(['Balanced Accuracy: ', num2str(balanced_accuracy*100), '%']);
        disp(['F1 Score: ', num2str(f1_score)]);
        disp(['AUC: ', num2str(auc)]);
        disp('Confusion Matrix:');
        disp(confusion_mat_val);
    end
    
    % 7. Find optimal threshold using Youden's J statistic
    [~, optimal_idx] = max(tpr - fpr);
    optimal_threshold = (min(val_pred_prob) + max(val_pred_prob)) / 2; % Can be refined
    
    % 8. Make predictions on test set with optimal threshold
    if exist('X_test', 'var')
        test_pred_prob = predict(model, X_test);
        test_pred_class = double(test_pred_prob >= optimal_threshold);
        
        % Output probabilities and classes
        %disp('Test set predictions:');
        %disp([test_pred_prob, test_pred_class]);
    end
end