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
    perturbed_values=[];%记录每个变量扰动后的函数值，即论文中的向量h，节省FEs（可以考虑在后续算法设计中使用类似表示历史信息的向量）

    %% 找到所有的可分离变量
    p1 = base * ones(1,dim);% 设为下限值
    fp1 = feval(fun, p1, fun_number);% 计算p1函数值
    p4 = (base+sigma) * ones(1,dim);% 设为上限值
    fp4 = feval(fun, p4,fun_number);% 计算p4函数值

    FEs = FEs + 2;% 评估次数加2

    for i =1:dim
        p2=p1;% 设为下限值
        p2(i)=(base+sigma);% 第i维设为上限值
        fp2=feval(fun, p2, fun_number);% 计算p2函数值
        
        p3=p4;% 设为上限值
        p3(i)=base;% 第i维设为下限值
        fp3=feval(fun, p3, fun_number);% 计算p3函数值
        
        FEs = FEs +2;% 评估次数加2

        delta1=fp2-fp1;
        delta2=fp4-fp3;
        perturbed_values=[perturbed_values;fp2];% 记录扰动后的函数值
        % 这里第i个数值的含义为除了第i维设为上限值外，其余维度全部设为下限值

        
%       adaptive epsilon
        epsilon = epsilonCalculate(fp1,fp2,fp3,fp4,dim); % 阈值计算函数（这里可以更换，考虑换为自适应阈值）
        if(abs(delta1-delta2) < epsilon)% 比较差值与阈值大小，判断是否为可分离变量            
            seps = [seps ;i];% 记录可分离变量
        else
            dims=[dims;i];% 记录不可分离变量
        end
    end
    
   %% 将不可分离变量分组
     if(length(dims)>1)
        [groups,~,FE]=mergeGroup(fun,fun_number,options,dims,fp1,perturbed_values);
        % perturbed_values里面为包含了所有扰动后fp2的值，所以没有传入fp2，重用历史信息
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
