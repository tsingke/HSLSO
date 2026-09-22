function [gbestX,gbestfitness,gbesthistory]=GTDE_tag(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,Func,FuncId)

%Gene Targeting Differential Evolution (GTDE)

FEs=0;
MaxFEs=10000*dimension;
if dimension>=500  %CEC2010测试集定义1000维(CEC 2013 F13,F14函数是905维)问题的MaxFEs=3E6（阈值自己定，只要检测出是大规模问题就行）
MaxFEs=3E6;
end
gbestfitness=inf;

x=rand(popsize,dimension); %位置向量
v=rand(popsize,dimension); %变异向量
u=rand(popsize,dimension); %试验向量


% 适应度空间分配
fitnessx= rand(1,popsize); % 群体适应度

Pm=0.01;
NGT=400;

ComputeFitness=Func;

% 加载变量分组结果（ERDG）
% 暂时只测cec2013
groupFilePath = strcat('./','CEC2013LSGO','/GroupResult/', 'ERDG');
groupFile = strcat(groupFilePath, '/f', num2str(FuncId), '_groups.mat');
load(groupFile, '-mat', 'groups', 'fEvalNum');    

FEs = FEs+fEvalNum;

% 对得到的分组结果进行处理
groups = preparationgroups(groups);

% 组数
groupN = size(groups,1);
deltaFit = zeros(groupN,1); % 用来保存变量组贡献度

%% 种群初始化
for i =1:popsize
    x(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    fitnessx(i)= ComputeFitness(x(i,:)',FuncId); % 个体适应度
    FEs=FEs+1;
    if gbestfitness>fitnessx(i)
        gbestfitness=fitnessx(i);
        gbestX= x(i,:);
        idx = i;
    end
    gbesthistory(FEs)=gbestfitness;
    fprintf("GTDE 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
end

for m = 1:groupN
    dims = groups{m};
    gbestfitnessbefore = gbestfitness;
    % 子维度组贡献值初始化
    for k=1:NGT   %算法1
        Pj=normrnd(ones(1,length(dims))*0.01,ones(1,length(dims))*0.01);
        bottleneck=rand(1,length(dims))<Pj;  %返回逻辑值，用1标记瓶颈维度，0表示非瓶颈维度
        F=normrnd(0.5,0.1);
        r=[];
        r=selectID(popsize,i,2);
        r1=r(1);
        r2=r(2);
        
        for j=1:length(dims) %算法2
            if rand<Pm
                xrand=xmin+(xmax-xmin)*rand;
                v(idx,dims(j))=gbestX(dims(j))+F*(x(r1,dims(j))-xrand);
            else
                v(idx,dims(j))= gbestX(dims(j))+F*(x(r1,dims(j))-x(r2,dims(j)));
            end
        end
        
        Flag4ub=v(idx,:)>xmax;
        Flag4lb=v(idx,:)<xmin;
        v(idx,:)=(v(idx,:).*(~(Flag4ub+Flag4lb)))+(xmin+(xmax-xmin)*rand(1,dimension)).*Flag4ub+(xmin+(xmax-xmin)*rand(1,dimension)).*Flag4lb;
        
        newgbestX=gbestX;
        newgbestX(dims(bottleneck))=v(idx,dims(bottleneck));  %算法3
        newgbestfitness=ComputeFitness(newgbestX',FuncId);
        FEs=FEs+1;
        if newgbestfitness<=gbestfitness
            gbestfitness=newgbestfitness;
            gbestX=newgbestX;
            x(idx,:)=gbestX; %我觉得还有这句，不然全局最优不会放入种群中，造成x(i,:)和gbestX很难相等(对全局最优的操作将很难进行)
        end
        gbesthistory(FEs)=gbestfitness;
        fprintf("GTDE 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
    end
    deltaFit(m) = gbestfitnessbefore - gbestfitness;

end

while 1
    for i =1:popsize
        
        if isequal(x(i,:),gbestX)  %针对最佳个体(如果i是最佳个体)

            % 选择进行靶向的分组
            [~,mid] = max(deltaFit); 
            dims = groups{mid};
            gbestfitnessbefore = gbestfitness;
            
            for k=1:NGT   %算法1
                Pj=normrnd(ones(1,length(dims))*0.01,ones(1,length(dims))*0.01);
                bottleneck=rand(1,length(dims))<Pj;  %返回逻辑值，用1标记瓶颈维度，0表示非瓶颈维度
                F=normrnd(0.5,0.1);
                r=[];
                r=selectID(popsize,i,2);
                r1=r(1);
                r2=r(2);
                
                for j=1:length(dims) %算法2
                    if rand<Pm
                        xrand=xmin+(xmax-xmin)*rand;
                        v(idx,dims(j))=gbestX(dims(j))+F*(x(r1,dims(j))-xrand);
                    else
                        v(idx,dims(j))= gbestX(dims(j))+F*(x(r1,dims(j))-x(r2,dims(j)));
                    end
                end
                
                Flag4ub=v(i,:)>xmax;
                Flag4lb=v(i,:)<xmin;
                v(i,:)=(v(i,:).*(~(Flag4ub+Flag4lb)))+(xmin+(xmax-xmin)*rand(1,dimension)).*Flag4ub+(xmin+(xmax-xmin)*rand(1,dimension)).*Flag4lb;
                
                newgbestX=gbestX;
                newgbestX(dims(bottleneck))=v(i,dims(bottleneck));  %算法3
                newgbestfitness=ComputeFitness(newgbestX',FuncId);
                FEs=FEs+1;
                if newgbestfitness<=gbestfitness
                    gbestfitness=newgbestfitness;
                    gbestX=newgbestX;
                    x(i,:)=gbestX; %我觉得还有这句，不然全局最优不会放入种群中，造成x(i,:)和gbestX很难相等(对全局最优的操作将很难进行)
                end
                gbesthistory(FEs)=gbestfitness;
                fprintf("GTDE 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
            end

            % 计算所选分组对适应度的贡献值
            deltaFit(mid) = gbestfitnessbefore - gbestfitness;
            
        else  %其他个体
            
            F = normrnd(0.7,0.5);
            CR = normrnd(0.5,0.5);
            r=[];
            r=selectID(popsize,i,2);
            r1=r(1);
            r2=r(2);
            v(i,:)=x(i,:)+F*(gbestX - x(i,:))+ F*(x(r1,:) - x(r2,:));
            jrand =randi([1,popsize],1);
            for j =1:dimension
                if(rand <= CR || j==jrand)
                    u(i,j)=v(i,j);
                else
                    u(i,j)=x(i,j);
                end
            end
            
            Flag4ub=u(i,:)>xmax;
            Flag4lb=u(i,:)<xmin;
            u(i,:)=(u(i,:).*(~(Flag4ub+Flag4lb)))+(xmin+(xmax-xmin)*rand(1,dimension)).*Flag4ub+(xmin+(xmax-xmin)*rand(1,dimension)).*Flag4lb;
            
            ufitness = ComputeFitness(u(i,:)',FuncId);
            FEs=FEs+1;
            if ufitness <= fitnessx(i)
                x(i,:) = u(i,:);
                fitnessx(i)=ufitness;
            end
            
            if gbestfitness>fitnessx(i)
                gbestfitness=fitnessx(i);
                gbestX= x(i,:);
            end
            gbesthistory(FEs)=gbestfitness;
            fprintf("GTDE 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
        end
    end
    
    if FEs>=MaxFEs
        break;
    end
    
    
end

if FEs<MaxFEs
    gbesthistory(FEs+1:MaxFEs)=gbestfitness;
else
    if FEs>MaxFEs
        gbesthistory(MaxFEs+1:end)=[];
    end
end
end % end function

%% ---------------------------------------------------------------------

%% 选择不同的函数
function [r]=selectID(popsize,i,count)
% 函数功能：在[1,popsizze]内随机生成count个不包括i的彼此不重复的整数值
% 函数返回： 列向量r，向量的维度为count。
% 函数思想： 把已经选择的元素，从数组里去除。
if count<= popsize
    %1.除去i的值，生成新的向量vec
    vec=[1:i-1,i+1:popsize];
    
    %2.随机生成count个不一样的数值
    r=zeros(1,count);
    
    for j =1:count
        n = popsize-j;   % 当前vec中向量的个数
        t = randi(n,1,1);% 产生一个随机整数
        r(j) = vec(t);   % 取随机数
        vec(t)=[]; %从数组中删除当前元素,防止某个数再被选择
    end
end
end

%% 分组结果预处理
function groups = preparationgroups(groups)
m = size(groups,1);

if m == 1000
    % 对于完全可分离函数
    matgroups = cell2mat(groups);
    length = size(matgroups,1);
    groups = {};
    a = 1;
    g = 0;
    while a <= length
        
        temgroups = [];
        s = 0;
        for s = 1:50
            temgroups = [temgroups,matgroups(a)];
            a = a+1;
        end
        g = g+1;
        groups{g} = temgroups;
    end
    groups = groups';

elseif m == 1
    % 对于完全不可分离函数
    matgroups = cell2mat(groups);
    length = size(matgroups,2);
    groups = {};
    a = 1;
    g = 0;
    while a <= length
        
        temgroups = [];
        s = 0;
        for s = 1:50
            temgroups = [temgroups,matgroups(a)];
            a = a+1;
        end
        g = g+1;
        groups{g} = temgroups;
    end
    groups = groups';

elseif m > 200
    % 对于完全可分离变量较多的部分可分离函数
    matgroups = [];
    g = 0;
    for i = 1:m
        if size(groups{i})==1
            matgroups = [matgroups,groups{i}];
        else
            g = g+1;
        end
    end
    length = size(matgroups,2);
    a = 1;
    while a <= length        
        temgroups = [];
        s = 0;
        for s = 1:50
            temgroups = [temgroups,matgroups(a)];
            a = a+1;
        end
        g = g+1;
        groups{g} = temgroups;
    end
    groups = groups(1:g);
end
end

