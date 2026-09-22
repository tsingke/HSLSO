% The SLPSO optimizer
% Reference:
% Cheng R, Jin Y. A social learning particle swarm optimization algorithm 
% for scalable optimization[J]. Information Sciences, 2015, 291: 43-60.

function [pop,fitness,bestmem,bestval,v] = slpso(Func,func_num,dim_index,pop,fitness,bestmem,bestval,Lbound,Ubound,FEmaxCycle,v)

    
    [NP, d] = size(pop);
    M = 100;
    PL = zeros(NP,1);
    c3 = d/M*0.01;
    for i = 1 : NP
        PL(i) = (1 - (i - 1)/NP)^log(sqrt(ceil(d/NP)));
    end
    
    FEs = 0;
    while(FEs < FEmaxCycle)
        % population sorting
        [fitness rank] = sort(fitness, 'descend');
        pop = pop(rank,:);
        v = v(rank,:);
        
        % center position
        center = ones(NP,1)*mean(pop);
        
        %random matrix 
        %rand('seed', sum(100 * clock));
        randco1 = rand(NP, d);
        %rand('seed', sum(100 * clock));
        randco2 = rand(NP, d);
        %rand('seed', sum(100 * clock));
        randco3 = rand(NP, d);
        winidxmask = repmat([1:NP]', [1 d]);
        winidx = winidxmask + ceil(rand(NP, d).*(NP - winidxmask));
        pwin = pop;
        for j = 1:d
                pwin(:,j) = pop(winidx(:,j),j);
        end
        
        % social learning
         lpmask = repmat(rand(NP,1) < PL, [1 d]);
         lpmask(NP,:) = 0;
         v1 =  1*(randco1.*v + randco2.*(pwin - pop) + c3*randco3.*(center - pop));
         p1 =  pop + v1;   
              
         v = lpmask.*v1 + (~lpmask).*v;         
         pop = lpmask.*p1 + (~lpmask).*pop;
         
         % boundary control
        for i = 1:NP - 1
            pop(i,:) = max(pop(i,:), Lbound(i,:));
            pop(i,:) = min(pop(i,:), Ubound(i,:));
        end
              
        % fitness evaluation
        gpop = ones(NP, 1) * bestmem;
        gpop(:, dim_index) = pop;
        for i = 1:NP
            fitness(i) = Func(gpop(i,:)', func_num); 
        end
        [bestval, ibest] = min(fitness);
        bestmem = gpop(ibest, :);         
        FEs = FEs + NP;
    end
end