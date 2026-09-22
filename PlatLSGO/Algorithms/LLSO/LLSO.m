function [gbestX, gbestfitness, gbesthistory] = LLSO(mainHandle, PopSize, D, xmax, xmin, vmax, vmin, MaxIter, fCalculation, FuncId, VisualSwitch)

PopSize = 500;
phi = 0.4;
level_num = 10;
MaxFEs = 3e6;

FEs = 0;
gbesthistory = zeros(1, MaxFEs);

positions = xmin + (xmax - xmin) .* rand(PopSize, D);
speeds = zeros(PopSize, D);

fitness = zeros(1, PopSize);
for i = 1:PopSize
    fitness(i) = fCalculation(positions(i, :)', FuncId);
    FEs = FEs + 1;
end

[gbestfitness, gbest_idx] = min(fitness);
gbestX = positions(gbest_idx, :);
gbesthistory(1:FEs) = gbestfitness;

while FEs < MaxFEs

    [~, sorted_indices] = sort(fitness);

    level_size = floor(PopSize / level_num);

    levels = cell(1, level_num);
    start_idx = 1;
    for i = 1:(level_num - 1)
        levels{i} = sorted_indices(start_idx : start_idx + level_size - 1);
        start_idx = start_idx + level_size;
    end
    levels{level_num} = sorted_indices(start_idx : end);

    for i = level_num:-1:3
        current_level_indices = levels{i};

        for j = 1:length(current_level_indices)
            particle_idx = current_level_indices(j);

            rand_levels = randperm(i - 1, 2);
            rl1 = min(rand_levels);
            rl2 = max(rand_levels);

            exemplar1_idx = levels{rl1}(randi(length(levels{rl1})));
            exemplar2_idx = levels{rl2}(randi(length(levels{rl2})));

            exemplar1 = positions(exemplar1_idx, :);
            exemplar2 = positions(exemplar2_idx, :);

            r1 = rand(1, D);
            r2 = rand(1, D);
            r3 = rand(1, D);

            speeds(particle_idx, :) = r1 .* speeds(particle_idx, :) ...
                                    + r2 .* (exemplar1 - positions(particle_idx, :)) ...
                                    + phi * r3 .* (exemplar2 - positions(particle_idx, :));

            positions(particle_idx, :) = positions(particle_idx, :) + speeds(particle_idx, :);

            positions(particle_idx, :) = max(positions(particle_idx, :), xmin);
            positions(particle_idx, :) = min(positions(particle_idx, :), xmax);

            if FEs >= MaxFEs
                break;
            end

            fitness(particle_idx) = fCalculation(positions(particle_idx, :)', FuncId);
            FEs = FEs + 1;

            if fitness(particle_idx) < gbestfitness
                gbestfitness = fitness(particle_idx);
                gbestX = positions(particle_idx, :);
            end

            gbesthistory(FEs) = gbestfitness;

            if mod(FEs, floor(MaxFEs/10)) == 0
                fprintf('LLSO 第%d次评价，最佳适应度 = %e\n', FEs, gbestfitness);
            end
        end

        if FEs >= MaxFEs
            break;
        end
    end

    if FEs >= MaxFEs
        break;
    end

    if level_num > 1
        level_2_indices = levels{2};

        for j = 1:length(level_2_indices)
            particle_idx = level_2_indices(j);

            rand_exemplars = randperm(length(levels{1}), 2);
            p1_idx = levels{1}(rand_exemplars(1));
            p2_idx = levels{1}(rand_exemplars(2));

            if fitness(p2_idx) < fitness(p1_idx)
                [p1_idx, p2_idx] = deal(p2_idx, p1_idx);
            end

            exemplar1 = positions(p1_idx, :);
            exemplar2 = positions(p2_idx, :);

            r1 = rand(1, D);
            r2 = rand(1, D);
            r3 = rand(1, D);

            speeds(particle_idx, :) = r1 .* speeds(particle_idx, :) ...
                                    + r2 .* (exemplar1 - positions(particle_idx, :)) ...
                                    + phi * r3 .* (exemplar2 - positions(particle_idx, :));

            positions(particle_idx, :) = positions(particle_idx, :) + speeds(particle_idx, :);

            positions(particle_idx, :) = max(positions(particle_idx, :), xmin);
            positions(particle_idx, :) = min(positions(particle_idx, :), xmax);

            if FEs >= MaxFEs
                break;
            end

            fitness(particle_idx) = fCalculation(positions(particle_idx, :)', FuncId);
            FEs = FEs + 1;

            if fitness(particle_idx) < gbestfitness
                gbestfitness = fitness(particle_idx);
                gbestX = positions(particle_idx, :);
            end

            gbesthistory(FEs) = gbestfitness;

            if mod(FEs, floor(MaxFEs/10)) == 0
                fprintf('LLSO 第%d次评价，最佳适应度 = %e\n', FEs, gbestfitness);
            end
        end
    end
end

if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = gbestfitness;
end

end
