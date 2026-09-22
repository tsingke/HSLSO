function [gbestx,gbestfitness,gbesthistory]=D_CSO(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)
MaxFEs = 3e6;
FEs = 0;
num_initial = 500;
num_vari = dimension;
varphi = 0.1; % 100-D: 0, 500-D: 0.1 and 0.05, 1000-D: 0.15 and 0.1
func_num = FuncId;
benchmark_func = fCalculation;

sample_x = lhsdesign(num_initial, num_vari).*(xmax - xmin) + xmin.*ones(num_initial, num_vari);
for i = 1:num_initial
    sample_y(i,:) = benchmark_func(sample_x(i,:)',func_num);
end
FEs = FEs + num_initial;
v = zeros(num_initial,num_vari);  % v is initialized to 0
fmin = min(sample_y);
gbesthistory(1:FEs) = fmin;

while FEs <= MaxFEs

    % Partition pb and pw according to the swarm entropy; the entropy is computed similarly to MATLAB's entropy
    n_bin = num_initial;    % n_bin is the number of bins
    res = rescale(sample_y);    % rescale scales the entries of the array to the interval [0,1]
    p = imhist(res,n_bin);  % rescale scales the entries of the array to the interval [0,1]
    p(p==0) = [];   % Remove the zeros in p
    p = p ./ numel(sample_y);% Normalize p so that sum(p) is 1; numel returns the number of elements in the array

    % So that the maximum of E is 1, logn(p) is used instead of log2(p)
    n_nozero = size(p,1);
    E = -sum(p.*(log(p)./log(n_nozero)));
    
    % Partition pb and pw
    [sample_y,ind]= sort(sample_y); % Sort the elements of A in ascending order
    sample_x = sample_x(ind,:);
    v = v(ind,:);
    mean_x = mean(sample_x);
    d = 0.45;   % 100-D: 0.25, 500-D: 0.35, 1000-D: 0.45
    if E > 1 - d
        mb = -(num_initial / d) * E + num_initial / d;
    else
        mb = num_initial;
    end
    mb = ceil(mb);
    mw = num_initial - mb;
    pb = sample_x(1:mb,:);  % Smaller is better
    pby = sample_y(1:mb,:);
    pbv = v(1:mb,:);
    pw = sample_x((mb + 1):num_initial,:);
    pwv = v((mb + 1):num_initial,:);

    next_samplex = [];
    next_sampley = [];
    next_v = [];
    % Update pw
    for i = 1:size(pw,1)
        if size(pb,1) == 0  % pb may be empty
            xj = sample_x(1,:);   % Current minimum
        else
            xj = pb(randi(size(pb,1)),:);
        end
        pwv(i,:) = rand(1, num_vari) .* pwv(i,:) + rand(1, num_vari) .* (xj - pw(i,:)) + varphi .* rand(1, num_vari) .* (mean_x - pw(i,:));
        xi = pw(i,:) + pwv(i,:);
        xi(xi > xmax) = xmax;
        xi(xi < xmin) = xmin;% Range check
        if FEs <= MaxFEs
            yi = benchmark_func(xi',func_num);
            if yi < fmin
                fmin = yi;
                gbestx = xi;
            end
            FEs = FEs + 1;
            gbesthistory(FEs) = fmin;
            if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                fprintf("DCSO  FE %d  best = %e\n",FEs,fmin);
            end
            next_samplex = [next_samplex;xi];   % Add to the next iteration
            next_sampley = [next_sampley;yi];
            next_v = [next_v;pwv(i,:)];
        else
            break;
        end
    end

    % Update pb
    while ~isempty(pb)
        if size(pb,1) == 1
            next_samplex = [next_samplex;pb];
            next_sampley = [next_sampley;pby];
            next_v = [next_v;pbv];
            pb = [];
            pby = [];
            pbv = [];
        else
            k1 = randi(size(pb, 1));    % Randomly draw two points
            k2 = randi(size(pb, 1));
            while k2 == k1
                k2 = randi(size(pb, 1));
            end
            if pby(k1,1) < pby(k2,1)
                add = k1;
                update = k2;
            else
                add = k2;
                update = k1;
            end
            next_samplex = [next_samplex;pb(add,:)]; % add is taken directly
            next_sampley = [next_sampley;pby(add,:)];
            next_v = [next_v;pbv(add,:)];
            pbv(update,:) = rand(1, num_vari) .* pbv(update,:) + rand(1, num_vari) .* (pb(add,:) - pb(update,:));    % update needs updating
            xl = pb(update,:) + pbv(update,:);
            xl(xl > xmax) = xmax;
            xl(xl < xmin) = xmin;% Range check
            if FEs <= MaxFEs
                yl = benchmark_func(xl',func_num);
                if yl < fmin
                    fmin = yl;
                    gbestx = xl;
                end
                FEs = FEs + 1;
                gbesthistory(FEs) = fmin;
                if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                    fprintf("DCSO  FE %d  best = %e\n",FEs,fmin);
                end
                next_samplex = [next_samplex;xl];   % Add to the next iteration
                next_sampley = [next_sampley;yl];
                next_v = [next_v;pbv(update,:)];
            else
                break;
            end

            % Remove k1 and k2 from pb; delete from the back so that the indices of the remaining deletions stay valid
            if k1 > k2
                first = k1;
                second = k2;
            else
                first = k2;
                second = k1;
            end
            pb(first, :) = [];
            pby(first, :) = [];
            pbv(first, :) = [];
            pb(second, :) = [];
            pby(second, :) = [];
            pbv(second, :) = [];
        end
    end
    
    % Update sample_x, v and sample_y
    if size(next_samplex,1) ~= num_initial   % Indicates that the program terminates
        break;
    else
        sample_x = next_samplex;
        sample_y = next_sampley;
        v = next_v;
    end
    gbestfitness = fmin;
end
if FEs<MaxFEs
    gbesthistory(FEs+1:MaxFEs)=gbestfitness;
else
    if FEs>MaxFEs
        gbesthistory(MaxFEs+1:end)=[];
    end
end
end