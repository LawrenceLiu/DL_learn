function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
% tic;
%for test
% lambda = 0;
% beta = 0;

datalength = size(data, 2);
% datalength = 100;
rou = ones(hiddenSize, 1) * sparsityParam;
% rou_hat = zeros(hiddenSize, 1);

% for i = 1 : datalength
%     a_1 = data(:, i);
%     z_2 = W1 * a_1 + b1;
%     a_2 = sigmoid(z_2);
%     %for sparse term
%     rou_hat = rou_hat + a_2;
% end
% rou_hat = rou_hat / datalength;

% for i = 1 : datalength
%     %forward
%     a_1 = data(:, i);
%     z_2 = W1 * a_1 + b1;
%     a_2 = sigmoid(z_2);
%     z_3 = W2 * a_2 + b2;
%     a_3 = sigmoid(z_3);
%     
%     %for squared error term
%     error_3 = (a_3 - a_1) .* (a_3 .* (1-a_3));
%     error_2 = (W2' * error_3 + beta * dKL(rou, rou_hat)) .* (a_2 .* (1-a_2));
%     W2grad = W2grad + error_3 * a_2';
%     b2grad = b2grad + error_3;
%     W1grad = W1grad + error_2 * a_1';
%     b1grad = b1grad + error_2;
%     %for cost
%     delta_3 = sum( 0.5 * ((a_1 - a_3).^2) );
%     cost = cost + delta_3;
% end
% W1grad = W1grad / datalength;
% b2grad = b2grad / datalength;
% W2grad = W2grad / datalength;
% b1grad = b1grad / datalength;
% cost = cost / datalength;

% vectorization, hey hey
%forward
% A_1 = data(:, 1:datalength); %%test
A_1 = data;
% B_1 = b1 * ones(1, datalength);
A_2 = sigmoid(W1 * A_1 + repmat(b1,1,datalength));
% B_2 = b2 * ones(1, datalength);
A_3 = sigmoid(W2 * A_2 + repmat(b2,1,datalength));

rou_hat = mean(A_2, 2);
error_3 = (A_3 - A_1) .* (A_3 .* (1-A_3));
error_2 = (W2' * error_3 + repmat(beta * dKL(rou, rou_hat),1,datalength) ) .* (A_2 .* (1-A_2));
W2grad = (error_3 * A_2') / datalength;
b2grad = mean(error_3, 2);
W1grad = (error_2 * A_1') / datalength;
b1grad = mean(error_2, 2);
delta_3 = sum( 0.5 * ((A_1 - A_3).^2) );
cost = mean(delta_3, 2);

%addweight decay
W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;
weight_term = sum(sum(W1.^2)) + sum(sum(W2.^2));
weight_term = weight_term * 0.5 * lambda;
cost = cost + weight_term;
%add sparsity penalty
KL = KL_divergence(rou, rou_hat);
cost = cost + beta * sum(KL);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
% toc;
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function dKL = dKL(rou, rou_hat)
% get derivative of KL_divergence
% rou, rou_hat are vectors, use scalar operation.
% -(rou/rou_hat) + (1-rou)/(1-rou_hat)
% (rou_hat - rou) / ((1-rou_hat)*rou_hat)
    dKL = (rou_hat - rou) ./ ((1-rou_hat).*rou_hat);
end

function KL = KL_divergence(rou, rou_hat)
% get the KL divergence
% rou, rou_hat are vectors, use scalar operation.
    t_rou = 1 - rou;
    t_rou_hat = 1 - rou_hat;
    KL = rou .* log(rou ./ rou_hat) + t_rou .* log(t_rou ./ t_rou_hat);
end
