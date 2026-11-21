function [gbestx,bestever,gbesthistory] = RLLPSO(sequence,a,lengthdata,L,maxiter,dimension,opts)
%RLLPSO MSA adaptation of the Q-learning level-based PSO algorithm.
% Q-learning, epsilon-greedy level selection, and level competition preserved.
% Maximizes internal SPS; includes initialization in its FE count.
if nargin<7,opts=struct();end
[popsize,MaxFEs]=msa_level_baseline_setup(maxiter,opts);
phi=0.4;alpha=0.4;gamma=0.8;epsilonRL=0.9;
S=[4,6,8,10,20,50];S=S(S<=floor(popsize/2));
if isempty(S),error('RLLPSO requires popsize >= 8.');end
ns=numel(S);Q=zeros(ns,ns);state=1;
X=rand(popsize,dimension);X(1,:)=0.5;
V=zeros(popsize,dimension);
fit=zeros(popsize,1);
feBest=zeros(MaxFEs,1);FEs=0;bestever=-inf;gbestx=[];
for i=1:popsize
    fit(i)=fitness(sequence,a,lengthdata,L,X(i,:));FEs=FEs+1;
    if fit(i)>bestever,bestever=fit(i);gbestx=X(i,:);end
    feBest(FEs)=bestever;
end
prevBest=bestever;
while FEs<MaxFEs
    if rand<epsilonRL
        [~,action]=max(Q(state,:));
    else
        action=randi(ns);
    end
    NL=S(action);
    [~,rankIndex]=sort(fit,'descend');
    ls=floor(popsize/NL);
    levels=cell(NL,1);
    for lev=1:NL-1,levels{lev}=rankIndex((lev-1)*ls+1:lev*ls);end
    levels{NL}=rankIndex((NL-1)*ls+1:end);
    Xnew=X;Vnew=V;
    for lev=2:NL
        for j=1:numel(levels{lev})
            k=levels{lev}(j);
            if lev==2
                pool=levels{1};
                e1=pool(randi(numel(pool)));e2=pool(randi(numel(pool)));
            else
                pComp=(FEs/MaxFEs)^2;
                l1=choose_level(lev,pComp);
                l2=choose_level(lev,pComp);
                if l2<l1,tmp=l1;l1=l2;l2=tmp;end
                pool1=levels{l1};pool2=levels{l2};
                e1=pool1(randi(numel(pool1)));
                e2=pool2(randi(numel(pool2)));
            end
            Vnew(k,:)=rand(1,dimension).*V(k,:) ...
                +rand(1,dimension).*(X(e1,:)-X(k,:)) ...
                +phi*rand(1,dimension).*(X(e2,:)-X(k,:));
            Vnew(k,:)=min(max(Vnew(k,:),0),1); % Source vmin=0,vmax=1
            Xnew(k,:)=min(max(X(k,:)+Vnew(k,:),0),1);
        end
    end
    % The algorithm evaluates every particle in each round, including
    % unchanged elite particles. This implementation preserves that policy.
    newfit=fit;
    for i=1:popsize
        if FEs>=MaxFEs,break;end
        newfit(i)=fitness(sequence,a,lengthdata,L,Xnew(i,:));FEs=FEs+1;
        if newfit(i)>bestever,bestever=newfit(i);gbestx=Xnew(i,:);end
        feBest(FEs)=bestever;
    end
    reward=abs(bestever-prevBest)/max(abs(prevBest),1e-12);
    prevBest=bestever;
    Q(state,action)=Q(state,action)+alpha*(reward+gamma*max(Q(action,:))-Q(state,action));
    state=action;
    X=Xnew;V=Vnew;fit=newfit;
end
gbesthistory=msa_level_baseline_history(feBest,maxiter);
end

function lev=choose_level(curL,probability)
% Source competition: with probability p, compare two distinct higher levels.
if curL<=2,lev=1;return;end
if rand<probability
    r=randperm(curL-1,2);lev=min(r);
else
    lev=randi(curL-1);
end
end
