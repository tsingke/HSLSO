% -------
% Inputs:
% -------
%    fun        : the function suite for which the interaction structure
%                 is going to be identified in this case benchmark_func
%                 of cec'2010 or cec'2013.
%
%    fun_number : the function number.
%
%    options    : this variable contains the options such as problem
%                 dimensionality, upper and lower bounds, and 
%                 parameters base and sigma used by merged differential grouping.
%
% --------
% Outputs:
% --------
%    sep      : a vector of all separable variables.
%    allgroups: a cell array containing all non-separable groups.
%    FEs      : the total number of fitness evaluations used.
%
% ------


function [seps, allgroups, FEs] = MDG(fun, fun_number, options)
    dim       = options.dim;
    base      = options.base;
    sigma     = options.sigma;
    dims      = [];%Save non-separable variables
    seps      = [];%Save separable variables
    allgroups = {};%save non-separable subcomponents
    FEs       = 0;
    perturbed_values=[];%Records the objective value after perturbing each variable, i.e. the vector h in the paper; saves FEs. (A similar history vector could be considered in future algorithm designs.)

    %% Find all the separable variables
    p1 = base * ones(1,dim);% Set to the lower bound
    fp1 = feval(fun, p1, fun_number);% Compute the function value of p1
    p4 = (base+sigma) * ones(1,dim);% Set to the upper bound
    fp4 = feval(fun, p4,fun_number);% Compute the function value of p4

    FEs = FEs + 2;% Increase the number of evaluations by 2

    for i =1:dim
        p2=p1;% Set to the lower bound
        p2(i)=(base+sigma);% Set the i-th dimension to the upper bound
        fp2=feval(fun, p2, fun_number);% Compute the function value of p2
        
        p3=p4;% Set to the upper bound
        p3(i)=base;% Set the i-th dimension to the lower bound
        fp3=feval(fun, p3, fun_number);% Compute the function value of p3
        
        FEs = FEs +2;% Increase the number of evaluations by 2

        delta1=fp2-fp1;
        delta2=fp4-fp3;
        perturbed_values=[perturbed_values;fp2];% Record the function value after perturbation
        % Here the i-th value means that apart from the i-th dimension set to the upper bound, all the other dimensions are set to the lower bound

        
%       adaptive epsilon
        epsilon = epsilonCalculate(fp1,fp2,fp3,fp4,dim); % Threshold computation function (it can be replaced here; consider switching to an adaptive threshold)
        if(abs(delta1-delta2) < epsilon)% Compare the difference with the threshold to decide whether the variable is separable            
            seps = [seps ;i];% Record the separable variable
        else
            dims=[dims;i];% Record the non-separable variable
        end
    end
    
   %% Group the non-separable variables
     if(length(dims)>1)
        [groups,~,FE]=mergeGroup(fun,fun_number,options,dims,fp1,perturbed_values);
        % perturbed_values contains all the perturbed fp2 values, so fp2 is not passed in, reusing the historical information
        FEs = FEs + FE;
        %Find groups with only one variable, and merge the variables into seps vector
        for i=size(groups,2):-1:1
            if(length(groups{i})==1)
                seps = [seps ;groups{i}];
                groups(i)=[];
            end
        end
        allgroups=groups;
     end
end
