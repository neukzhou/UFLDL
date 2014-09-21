function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------   

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
a1 = data;
z2 = bsxfun(@plus, W1*a1, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2*a2, b2);
a3 = z3;

m = size(a2);
rho = sparsityParam;
rhoi = sum(a2')' ./ m(1,2);
penalgrad = -bsxfun(@rdivide, rho, rhoi) + bsxfun(@rdivide, 1 - rho, 1 - rhoi);

delta3 = -(a1 - a3);
delta2 = bsxfun(@plus, W2' * delta3, beta * penalgrad) .* (a2 .* (1 - a2));

W1grad = W1grad + (delta2 * (a1')) / m(1, 2) + lambda * W1;
W2grad = W2grad + (delta3 * (a2')) / m(1, 2) + lambda * W2;
b1grad = b1grad + (sum(delta2')') / m(1, 2);
b2grad = b2grad + (sum(delta3')') / m(1, 2);

%W1 = W1 - 0.1 .* (W1grad ./ m(1, 2) + lambda .* W1);
%W1 = W2 - 0.1 .* (W2grad ./ m(1, 2) + lambda .* W2);
%b1 = b1 - 0.1 .* (b1grad ./ m(1, 2));
%b2 = b2 - 0.1 .* (b2grad ./ m(1, 2));

for i = 1:m(1, 2)
    cost = cost + norm((a3(:,i) - a1(:,i))) * norm((a3(:,i) - a1(:,i)));
end
cost = cost / 2 / m(1, 2) + lambda / 2 * (sum(sum(W1.*W1)) + sum(sum(W2.*W2))) ...
    + beta * sum(rho * log(rho ./ rhoi) + (1 - rho) * log((1 - rho) ./ (1 - rhoi)));











%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
