function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

X_norm = featureNormalize(X(:,2:size(X,2)));

indices = 1:size(X_norm,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = theta(1)
    
    for i = indices,
      x=X_norm(:,i);  % load data from x(i)
      h = h + (theta(i) * x); %calculate hyphotesis
    end;
    
    for i = indices,
      x=X_norm(:,i);  % load data from x(i)
      
      if i == 1,
        theta(1) = theta(1) - alpha * (1/m) * sum(h-y);
        theta(i+1) = theta(i+1) - alpha * (1/m) * sum((h-y) .* x);
      else
        theta(i+1) = theta(i+1) - alpha * (1/m) * sum((h-y) .* x);
      end;
    end;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
