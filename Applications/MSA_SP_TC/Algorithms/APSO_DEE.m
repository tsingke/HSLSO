function [gbestX,bestval,trace_val] = APSO_DEE(sequence,a,lengthdata,L,historyPoints,dim,MaxFEs)
% APSO-DEE MSA adaptation with an explicit FE budget.
% Initial population: 50. Group size follows eight FE-progress stages.
if nargin<7 || isempty(MaxFEs), error('APSO_DEE requires MaxFEs as input 7.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
Npop=50; ub=1; lb=0; phi=0.3; Nvar=dim; vmax=ub; vmin=-vmax;
if MaxFEs<Npop || MaxFEs~=floor(MaxFEs), error('Invalid MaxFEs.'); end

Position=lb+rand(Npop,Nvar)*(ub-lb);
Position(1,:)=0.5;
Velocity=zeros(Npop,Nvar);
Fitness=zeros(Npop,1);
feBest=nan(MaxFEs,1); fes=0; gbestfitness=-inf; gbestX=[];
for i=1:Npop
    Fitness(i)=fitness(sequence,a,lengthdata,L,Position(i,:));
    fes=fes+1;
    if Fitness(i)>gbestfitness, gbestfitness=Fitness(i); gbestX=Position(i,:); end
    feBest(fes)=gbestfitness;
end
[Fitness,initialOrder]=sort(Fitness,'descend');
Position=Position(initialOrder,:); Velocity=Velocity(initialOrder,:);

groupSchedule=[2 4 8 10 20 25 40 50];
gen=0; noProgress=0;
while fes<MaxFEs
    gen=gen+1;
    oldFEs=fes;
    stage=min(8,floor((fes/MaxFEs)*8)+1);
    groupsize=groupSchedule(stage);

    crowding=LSD_safe(Fitness);
    [~,index_crowding]=sort(crowding);
    [converg_group,groupnum]=Grouping(groupsize,Npop);
    losers=zeros(1,Npop); con_winners=zeros(1,Npop); div_winners=zeros(1,Npop);
    count=1;
    for cn=1:groupnum
        temp_list=converg_group(cn,:);
        temp_list(temp_list==0)=[];
        computing_size=length(temp_list);
        for k=2:computing_size
            pos_in_entropy=find(index_crowding==temp_list(k),1);
            if isempty(pos_in_entropy), continue; end
            if pos_in_entropy<=rand*Npop
                losers(count)=temp_list(k);
                con_winners(count)=temp_list(1);
                pick=pos_in_entropy+ceil(rand*(Npop-pos_in_entropy));
                pick=max(1,min(Npop,pick));
                div_winners(count)=index_crowding(pick);
                count=count+1;
            end
        end
    end
    losers(losers==0)=[]; con_winners(con_winners==0)=[]; div_winners(div_winners==0)=[];
    lenloser=min([numel(losers),numel(con_winners),numel(div_winners)]);
    losers=losers(1:lenloser); con_winners=con_winners(1:lenloser); div_winners=div_winners(1:lenloser);

    for k=1:lenloser
        if fes>=MaxFEs, break; end
        li=losers(k);
        w=rand(1,Nvar); r1=rand(1,Nvar); r2=rand(1,Nvar);
        Velocity(li,:)=w.*Velocity(li,:) ...
            +r1.*(Position(con_winners(k),:)-Position(li,:)) ...
            +phi*r2.*(Position(div_winners(k),:)-Position(li,:));
        Velocity(li,:)=max(Velocity(li,:),vmin);
        Velocity(li,:)=min(Velocity(li,:),vmax);
        Position(li,:)=Position(li,:)+Velocity(li,:);
        Position(li,:)=min(max(Position(li,:),lb),ub);
        Fitness(li)=fitness(sequence,a,lengthdata,L,Position(li,:));
        fes=fes+1;
        if Fitness(li)>gbestfitness, gbestfitness=Fitness(li); gbestX=Position(li,:); end
        feBest(fes)=gbestfitness;
    end

    [Fitness,index]=sort(Fitness,'descend');
    Position=Position(index,:); Velocity=Velocity(index,:);
    if Fitness(1)>gbestfitness, gbestfitness=Fitness(1); gbestX=Position(1,:); end

    if fes==oldFEs
        noProgress=noProgress+1;
    else
        noProgress=0;
    end
    if noProgress>1000
        error('APSO_DEE failed to consume FEs for 1000 consecutive generations.');
    end
    if mod(gen,25)==0 || fes>=MaxFEs
        fprintf('APSO-DEE: generation=%d, groupSize=%d, FEs=%d/%d, best SPS=%e\n', ...
            gen,groupsize,fes,MaxFEs,gbestfitness);
    end
end
bestval=gbestfitness;
trace_val=msa_level_baseline_history(feBest,historyPoints);
gbestX=gbestX(:)';
end

function [converg_group,groupnum]=Grouping(groupsize,Npop)
groupnum=ceil(Npop/groupsize);
converg_group=zeros(groupnum,groupsize);
init_index=1:Npop;
cn=1;
while cn<=groupnum-1
    pos=sort(randperm(numel(init_index),groupsize));
    converg_group(cn,:)=init_index(pos);
    init_index(pos)=[];
    cn=cn+1;
end
converg_group(cn,1:numel(init_index))=init_index;
end

function entropy=LSD_safe(fitness)
% Original LSD crowding indicator, with guards for degenerate/tied fitness.
len=length(fitness);
entropy=zeros(1,len);
if len<3
    return;
end
l=(fitness(3:len)-fitness(1:len-2));
den=fitness(end)-fitness(1);
if abs(den)<=eps || any(abs(l)<=eps)
    return;
end
L=[l(1) l l(end)];
L=L/den;
maxL=max(L);
if ~isfinite(maxL) || abs(maxL)<=eps
    return;
end
L=L/maxL;

l1=(fitness(2:len-1)-fitness(1:len-2))./l;
l1(2,:)=(fitness(3:len)-fitness(2:len-1))./l;
l2=sort(l1,1);
if any(abs(l2(2,:))<=eps)
    return;
end
x=l2(1,:)./l2(2,:);
entropy(1)=0;
entropy(end)=0;
entropy(2:len-1)=x;
maxEntropy=max(entropy);
if ~isfinite(maxEntropy) || abs(maxEntropy)<=eps
    entropy=zeros(1,len);
    return;
end
entropy=entropy/maxEntropy;
entropy=entropy.*L;
entropy(~isfinite(entropy))=0;
end
