function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
% softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

datalength = size(data, 2);
stackA = cell( [numel(stack)+2,1] );
stackA{1}.A = data;
soft_depth = numel(stackA);
for d = 2:soft_depth-1
    W = stack{d-1}.w;
    b = stack{d-1}.b;
    A = stackA{d-1}.A;
    stackA{d}.A = sigmoid(W * A + repmat(b,1,datalength));
end

% cost = 0; % You need to compute this
% You might find these variables useful
% M = size(data, 2);
% groundTruth = full(sparse(labels, 1:M, 1));
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1, numClasses, numCases));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
stackA{soft_depth}.A = softmaxTheta * stackA{soft_depth-1}.A;
stackA{soft_depth}.A = bsxfun(@minus, stackA{soft_depth}.A, max(stackA{soft_depth}.A, [], 1));
stackA{soft_depth}.A = exp(stackA{soft_depth}.A);
Prob = bsxfun(@rdivide, stackA{soft_depth}.A, sum(stackA{soft_depth}.A));
cost = -1 * groundTruth(:)' * log(Prob(:)) / numCases + sum(softmaxTheta(:).^2) * lambda/2;

% for softmax gradient
softmaxThetaGrad = -1 * (groundTruth - Prob) * stackA{soft_depth-1}.A' / numCases + lambda * softmaxTheta;
% for hidden layers
for d = soft_depth-1:-1:2
    if d == soft_depth -1 
        error_last = -1 * softmaxTheta' * (groundTruth - Prob) .* (stackA{d}.A .* (1 - stackA{d}.A));
    else
        W = stack{d}.w;
        b = stack{d}.b;  
        error_last = W' * error_last .* (stackA{d}.A .* (1 - stackA{d}.A));
    end    
    
    A = stackA{d-1}.A;
    stackgrad{d-1}.w = (error_last * A') / datalength;
    stackgrad{d-1}.b = mean(error_last, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
