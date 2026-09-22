function [gbestx,bestever,gbesthistory]=HCLPSO(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)


popsize = 500;
beta = 0.95; % The two parameters above were obtained through parameter sensitivity analysis
sigma = 0.1;
SPRmax = 0.9;
SPRmin = 0.45;

ComputeFitness = fCalculation;
FEs = 0;
MaxFEs = 3e6;

% Swarm initialization
for i =1:popsize
    p(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i)= ComputeFitness(p(i,:)',FuncId); % Individual fitness
end
FEs = FEs+popsize;

[bestever,id] = min(fitness);
gbestx = p(id,:); 

gbesthistory = [bestever*ones((FEs),1)]; 

gen = 1;

while FEs < MaxFEs
    [fitness,ind] = sort(fitness);
    p = p(ind,:); v = v(ind,:);
    SPR = SPRmax-(SPRmax-SPRmin)*((FEs/MaxFEs)^0.5);
    NSP = round(SPR*popsize);
    for i = 1:NSP
        w(i) = (1/(sigma*NSP*sqrt(2*pi)))*exp(-(((i-1)^2)/(2*sigma*sigma*NSP*NSP)));
    end
    pj = w./sum(w);

    for i = NSP+1:popsize
        % Select an elite individual; there should be a simpler function for this, but I do not know it ...
        r = rand;
        sp = 0;
        for k = 1:NSP
            r = r-pj(k);
            if r <= 0
                sp = k;
                break;
            end
        end

        v(i,:) = rand(1,dimension).*v(i,:)+rand(1,dimension).*beta.*(p(sp,:)-p(i,:));
        p(i,:) = p(i,:) + v(i,:);
        p(i,:) = max(p(i,:), xmin);
        p(i,:) = min(p(i,:), xmax);
        fitness(i) = ComputeFitness(p(i,:)', FuncId);
        FEs = FEs+1;
        if  fitness(i) < bestever
            bestever = fitness(i);
            gbestx = p(i,:);
        end

        gbesthistory(FEs) = bestever;
        if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
            fprintf("HCLPSO  FE %d  best = %e\n",FEs,bestever);
        end
    end

    for i = 1:NSP
        s = randperm(NSP,2);
        s(s==i) = [];
        s = s(1);
        if fitness(s)<fitness(i)
            v(i,:) = rand(1,dimension).*v(i,:)+rand(1,dimension).*beta.*(p(s,:)-p(i,:));
            p(i,:) = p(i,:) + v(i,:);
            p(i,:) = max(p(i,:), xmin);
            p(i,:) = min(p(i,:), xmax);
            fitness(i) = ComputeFitness(p(i,:)', FuncId);
            FEs = FEs+1;
            if  fitness(i) < bestever
                bestever = fitness(i);
                gbestx = p(i,:);
            end
    
            gbesthistory(FEs) = bestever;
            if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                fprintf("HCLPSO  FE %d  best = %e\n",FEs,bestever);
            end
        end
    end
    gen = gen+1;

end
    if FEs<MaxFEs
        gbesthistory(FEs+1:MaxFEs)=gbestfitness;
    else
        if FEs>MaxFEs
            gbesthistory(MaxFEs+1:end)=[];
        end
    end
end