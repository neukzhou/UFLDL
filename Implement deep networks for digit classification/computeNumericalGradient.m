function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

epsilon = 1e-4;

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

m = size(theta);

for i = 1:m(1, 1)
    thetaPlus = theta;
    thetaPlus(i) = thetaPlus(i) + epsilon;
    thetaMinus = theta;
    thetaMinus(i) = thetaMinus(i) - epsilon;
    [value1, grad1] = J(thetaPlus);
    [value2, grad2] = J(thetaMinus);
    numgrad(i) = (value1 - value2) / (2*epsilon);
end



%% ---------------------------------------------------------------
end
