function [gbestx,bestever,gbesthistory] = LLSO(sequence,a,lengthdata,L,maxiter,dimension,opts)
%LLSO MSA adaptation of the 2018 level-based learning algorithm.
% Maximizes Core/fitness.m (internal SPS), unlike the CEC minimization code.
% Particle budget, bounds, initialization, and output follow the MSA experiment configuration.
if nargin < 7, opts=struct(); end
[popsize,MaxFEs] = msa_level_baseline_setup(maxiter,opts);
phi=0.4; level_num=min(10,floor(popsize/2));
if level_num < 3, error('LLSO needs at least three levels.'); end
p=rand(popsize,dimension);
p(1,:)=0.5; % Same unperturbed Profile-HMM seed as HSLSO.
v=zeros(popsize,dimension);
fitnessx=zeros(popsize,1);
feBest=zeros(MaxFEs,1); FEs=0; bestever=-inf; gbestx=[];
for i=1:popsize
    fitnessx(i)=fitness(sequence,a,lengthdata,L,p(i,:)); FEs=FEs+1;
    if fitnessx(i)>bestever, bestever=fitnessx(i); gbestx=p(i,:); end
    feBest(FEs)=bestever;
end
while FEs<MaxFEs
    [~,order]=sort(fitnessx,'descend');
    ls=floor(popsize/level_num);
    levels=cell(1,level_num);
    for z=1:level_num-1
        levels{z}=order((z-1)*ls+1:z*ls);
    end
    levels{level_num}=order((level_num-1)*ls+1:end);
    for lev=level_num:-1:2
        members=levels{lev};
        for t=1:numel(members)
            if FEs>=MaxFEs, break; end
            k=members(t);
            if lev==2
                pick=randperm(numel(levels{1}),2);
                e1=levels{1}(pick(1)); e2=levels{1}(pick(2));
                if fitnessx(e2)>fitnessx(e1), tmp=e1;e1=e2;e2=tmp;end
            else
                selected=sort(randperm(lev-1,2));
                group1=levels{selected(1)};group2=levels{selected(2)};
                e1=group1(randi(numel(group1)));
                e2=group2(randi(numel(group2)));
            end
            v(k,:)=rand(1,dimension).*v(k,:) ...
                 +rand(1,dimension).*(p(e1,:)-p(k,:)) ...
                 +phi*rand(1,dimension).*(p(e2,:)-p(k,:));
            p(k,:)=min(max(p(k,:)+v(k,:),0),1);
            fitnessx(k)=fitness(sequence,a,lengthdata,L,p(k,:));
            FEs=FEs+1;
            if fitnessx(k)>bestever, bestever=fitnessx(k); gbestx=p(k,:); end
            feBest(FEs)=bestever;
        end
        if FEs>=MaxFEs, break; end
    end
end
gbesthistory=msa_level_baseline_history(feBest,maxiter);
end
