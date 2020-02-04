function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% create a blank results matrix
results = zeros(length(C_list) * length(s_list), 3); 

row = 1;
for i=1:length(C_list)
    for j=1:length(s_list)

        model = svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, s_list(j)));
        predictions = svmPredict(model, Xval);
        err_val = mean(double(predictions ~= yval));
        % save the results in the matrix
        results(row,:) = [C_list(i) s_list(j) err_val];
       row = row + 1;
    endfor
endfor

% use the min() function on the results matrix to find 
%   the C and sigma values that give the lowest validation error

lowest_error = find(results(:,3) == min(results(:,3)));
lowest_set = results(lowest_error, :);
C = lowest_set(1);
sigma = lowest_set(2);

% =========================================================================

end
