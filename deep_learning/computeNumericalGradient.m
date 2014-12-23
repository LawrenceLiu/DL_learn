function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
EPSILON = 0.0001;
dimensions = size(theta, 1);
args = eye(dimensions) * EPSILON;

for i = 1 : dimensions
    fprintf('Parameter index: %ld\n', i);
    value_1 = J(theta + args(:,i));
    value_2 = J(theta - args(:,i)); 
    numgrad(i) = (value_1 - value_2 ) ./ (2 * EPSILON);
end

%% ---------------------------------------------------------------
end
