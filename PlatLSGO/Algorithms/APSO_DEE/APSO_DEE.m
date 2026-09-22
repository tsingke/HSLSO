function [gbestX,bestval,trace_val] = APSO_DEE(~,Npop,dim,ub,lb,vmax,vmin,maxiter, fCalculation,funcid,~)
%Adaptive particle swarm optimizer with decoupling balance for
%exploration and exploitation (APSO-DEE)
%Npop: the swarm size
%funcid: the function id
%phi: the exploration balance parameter
%groupsize: the exploitation balance parameter which is adjusted by an
%adaptive strategy in APSO-DEE
Npop=1000;
MaxFEs = 1e4*dim;
if dim>500
    MaxFEs=3e6;
end
phi = 0.3;
Nvar = dim;
benchmark_func = fCalculation;
vmax = ub;
vmin = lb;
gen = 1;
gbestfitness=inf;
[bestval,trace_val,trace_std,Position,Velocity,Fitness,fes] = initialization(lb,ub,Npop,Nvar,fCalculation,funcid);
[tempfitness,~]=min(Fitness);
if tempfitness<=gbestfitness
    gbestfitness=tempfitness;
end
trace_val(1:fes) = gbestfitness;

mean_p = repmat(mean(Position),Npop,1);
trace_std(gen) = sum(sqrt(sum((Position-mean_p).*(Position-mean_p),2)))/Npop;
while fes <= MaxFEs
    if fes <= 375000
        groupsize = 2;
    elseif fes <= 375000*2
        groupsize = 4;
    elseif fes <= 375000*3
        groupsize = 8;
    elseif fes <= 375000*4
        groupsize = 10;
    elseif fes <= 375000*5
        groupsize = 20;
    elseif fes <= 375000*6
        groupsize = 25;
    elseif fes <= 375000*7
        groupsize = 40;
    else
        groupsize = 50;
    end
    crowding = LSD(Fitness);
    [~, index_crowding] = sort(crowding);
    [converg_group, groupnum] = Grouping(groupsize, Npop);
    losers = zeros(1,Npop);
    con_winners = zeros(1,Npop);
    div_winners = zeros(1,Npop);
    cn = 1;
    count = 1;
    while cn <= groupnum
        temp_list = converg_group(cn, :);
        temp_list(temp_list == 0) = [];
        computing_size = length(temp_list);
        k = 2;
        while k <= computing_size
            pos_in_entropy = find(index_crowding == temp_list(k));
            if  pos_in_entropy <= rand*Npop
                losers(count) = temp_list(k);
                con_winners(count) = temp_list(1);
                div_winners(count) = index_crowding(pos_in_entropy + ceil(rand*(Npop - pos_in_entropy)));
                count = count + 1;
            end
            k = k + 1;
        end
        cn = cn + 1;
    end
    losers(losers == 0) = [];
    con_winners(con_winners == 0) = [];
    div_winners(div_winners == 0) = [];
    lenloser = length(losers);
    con_exemplar = Position(con_winners,:);
    div_exemplar = Position(div_winners,:);
    w = rand(lenloser, Nvar);
    r1 = rand(lenloser, Nvar);
    r2 = rand(lenloser, Nvar);
    Velocity(losers,:) = w.*Velocity(losers,:) + r1.*(con_exemplar - Position(losers,:)) ...
        + phi*r2.*(div_exemplar - Position(losers,:));
    Velocity(Velocity > vmax) = vmax;
    Velocity(Velocity < vmin) = vmin;
    Position(losers,:) = Position(losers,:) + Velocity(losers,:);
    Position = FeasibleFunction(Position,lb,ub);
    Fitness(losers) = benchmark_func(Position(losers,:)',funcid);
    [Fitness, index] = sort(Fitness);
    Position = Position(index,:);
    Velocity = Velocity(index,:);
    [tempfitness,id]=min(Fitness);
    if tempfitness<=gbestfitness
        gbestfitness=tempfitness;
    end
    fes=fes+lenloser;
    trace_val(fes-lenloser+1:fes) = gbestfitness;
 
    gbestX = Position(id,:);
  
    %         fprintf(['APSO-DEE: The best and FEs of Function ', num2str(funcid), ' (', num2str(run), '):%e'],bestval)
    if mod(fes, floor(MaxFEs/10)) == 0 && fes <= MaxFEs
        fprintf("APSO-DEE 第%d次评价，最佳适应度 = %e\n",fes,gbestfitness);
    end
    gen = gen + 1;

    mean_p = repmat(mean(Position),Npop,1);
    trace_std(gen) = sum(sqrt(sum((Position-mean_p).*(Position-mean_p),2)))/Npop;
end
if fes<MaxFEs
    trace_val(fes+1:MaxFEs)=gbestfitness;
else
    if fes>MaxFEs
        trace_val(MaxFEs+1:end)=[];
    end
end
end

% 初始化函数
function [bestval,trace_val,trace_std,Position,Velocity,Fitness,fes] = initialization(lb,ub,Npop,Nvar,fCalculation,funcid)
%Parameter initialization
%     MaxFEs = 3000000;
benchmark_func = fCalculation;
trace_val = [];
trace_std = [];
Position = lb + rand(Npop, Nvar)*(ub - lb);
Velocity = zeros(Npop, Nvar);
%     Fitness = benchmark_func(Position',funcid);
for i = 1:Npop
    Fitness(i) = benchmark_func(Position(i,:)',funcid);
end
[Fitness, index] = sort(Fitness);
Position = Position(index,:);
Velocity = Velocity(index,:);
bestval = min(Fitness);
fes = Npop;
end

function [converg_group,groupnum] = Grouping(groupsize, Npop)
groupnum = ceil(Npop/groupsize);
converg_group = zeros(groupnum, groupsize);
init_index = 1 : Npop;
cn = 1;
while cn <= groupnum - 1
    len = length(init_index);
    pos = sort(randperm(len,groupsize));
    converg_group(cn,:) = init_index(pos);
    init_index(pos) = [];
    cn = cn + 1;
end
work_list = length(init_index);
converg_group(cn,1:work_list) = init_index;
end

function [Position] = FeasibleFunction(Position,lb,ub)
%FEASIBLEFUNCTION 此处显示有关此函数的摘要
%检查候选解的是否在定义范围之内
Position(Position > ub) = ub;
Position(Position < lb) = lb;
end

function entropy = LSD(fitness)
len = length(fitness);
entropy = zeros(1, len);
l = (fitness(3:len)-fitness(1:len-2));
L = [l(1) l l(end)];
L = L/(fitness(end) - fitness(1));
L = L/max(L);
l1 = (fitness(2:len-1)-fitness(1:len-2))./l;
l1(2,:) = (fitness(3:len)-fitness(2:len-1))./l;
l2 = sort(l1);
x = l2(1,:)./l2(2,:);
entropy(1) = 0;
entropy(end) = 0;
entropy(2:len-1) = x;
entropy = entropy/max(entropy);
entropy = entropy.*L;
end


