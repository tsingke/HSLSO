function [gbestX,gbestfitness,gbesthistory]=GTDE_tag(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,Func,FuncId)

%Gene Targeting Differential Evolution (GTDE)

FEs=0;
MaxFEs=10000*dimension;
if dimension>=500  %The CEC2010 suite defines a 1000-D problem (CEC 2013 functions F13 and F14 are 905-D), for which MaxFEs=3E6 (the threshold is chosen by the user, as long as the problem is detected as large-scale)
MaxFEs=3E6;
end
gbestfitness=inf;

x=rand(popsize,dimension); %position vector
v=rand(popsize,dimension); %mutation vector
u=rand(popsize,dimension); %trial vector


% Fitness space allocation
fitnessx= rand(1,popsize); % population fitness

Pm=0.01;
NGT=400;

ComputeFitness=Func;

% Load the variable grouping result (ERDG)
% Only cec2013 is tested for now
groupFilePath = strcat('./','CEC2013LSGO','/GroupResult/', 'ERDG');
groupFile = strcat(groupFilePath, '/f', num2str(FuncId), '_groups.mat');
load(groupFile, '-mat', 'groups', 'fEvalNum');    

FEs = FEs+fEvalNum;

% Process the obtained grouping result
groups = preparationgroups(groups);

% number of groups
groupN = size(groups,1);
deltaFit = zeros(groupN,1); % used to store the contribution of each variable group

%% Population initialization
for i =1:popsize
    x(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    fitnessx(i)= ComputeFitness(x(i,:)',FuncId); % individual fitness
    FEs=FEs+1;
    if gbestfitness>fitnessx(i)
        gbestfitness=fitnessx(i);
        gbestX= x(i,:);
        idx = i;
    end
    gbesthistory(FEs)=gbestfitness;
    fprintf("GTDE  FE %d  best = %e\n",FEs,gbestfitness);
end

for m = 1:groupN
    dims = groups{m};
    gbestfitnessbefore = gbestfitness;
    % Initialize the contribution value of the sub-dimension group
    for k=1:NGT   %Algorithm 1
        Pj=normrnd(ones(1,length(dims))*0.01,ones(1,length(dims))*0.01);
        bottleneck=rand(1,length(dims))<Pj;  %Returns logical values: 1 marks a bottleneck dimension, 0 marks a non-bottleneck dimension
        F=normrnd(0.5,0.1);
        r=[];
        r=selectID(popsize,i,2);
        r1=r(1);
        r2=r(2);
        
        for j=1:length(dims) %Algorithm 2
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
        newgbestX(dims(bottleneck))=v(idx,dims(bottleneck));  %Algorithm 3
        newgbestfitness=ComputeFitness(newgbestX',FuncId);
        FEs=FEs+1;
        if newgbestfitness<=gbestfitness
            gbestfitness=newgbestfitness;
            gbestX=newgbestX;
            x(idx,:)=gbestX; %I think this line is also needed, otherwise the global best is not placed into the population, which makes x(i,:) and gbestX rarely equal (and makes operations on the global best very difficult)
        end
        gbesthistory(FEs)=gbestfitness;
        fprintf("GTDE  FE %d  best = %e\n",FEs,gbestfitness);
    end
    deltaFit(m) = gbestfitnessbefore - gbestfitness;

end

while 1
    for i =1:popsize
        
        if isequal(x(i,:),gbestX)  %For the best individual (if i is the best individual)

            % Select the group to be targeted
            [~,mid] = max(deltaFit); 
            dims = groups{mid};
            gbestfitnessbefore = gbestfitness;
            
            for k=1:NGT   %Algorithm 1
                Pj=normrnd(ones(1,length(dims))*0.01,ones(1,length(dims))*0.01);
                bottleneck=rand(1,length(dims))<Pj;  %Returns logical values: 1 marks a bottleneck dimension, 0 marks a non-bottleneck dimension
                F=normrnd(0.5,0.1);
                r=[];
                r=selectID(popsize,i,2);
                r1=r(1);
                r2=r(2);
                
                for j=1:length(dims) %Algorithm 2
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
                newgbestX(dims(bottleneck))=v(i,dims(bottleneck));  %Algorithm 3
                newgbestfitness=ComputeFitness(newgbestX',FuncId);
                FEs=FEs+1;
                if newgbestfitness<=gbestfitness
                    gbestfitness=newgbestfitness;
                    gbestX=newgbestX;
                    x(i,:)=gbestX; %I think this line is also needed, otherwise the global best is not placed into the population, which makes x(i,:) and gbestX rarely equal (and makes operations on the global best very difficult)
                end
                gbesthistory(FEs)=gbestfitness;
                fprintf("GTDE  FE %d  best = %e\n",FEs,gbestfitness);
            end

            % Compute the fitness contribution of the selected group
            deltaFit(mid) = gbestfitnessbefore - gbestfitness;
            
        else  %Other individuals
            
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
            fprintf("GTDE  FE %d  best = %e\n",FEs,gbestfitness);
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

%% Select different functions
function [r]=selectID(popsize,i,count)
% Function: randomly generate count mutually distinct integers in [1,popsize] that do not include i
% Returns: a column vector r whose dimension is count.
% Idea: remove the already selected elements from the array.
if count<= popsize
    %1. Remove the value i and generate a new vector vec
    vec=[1:i-1,i+1:popsize];
    
    %2. Randomly generate count distinct values
    r=zeros(1,count);
    
    for j =1:count
        n = popsize-j;   % current number of elements in vec
        t = randi(n,1,1);% generate a random integer
        r(j) = vec(t);   % take the random value
        vec(t)=[]; %remove the current element from the array to prevent it from being selected again
    end
end
end

%% Grouping result preprocessing
function groups = preparationgroups(groups)
m = size(groups,1);

if m == 1000
    % For fully separable functions
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
    % For fully non-separable functions
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
    % For partially separable functions with many fully separable variables
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

