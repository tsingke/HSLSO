clc;
clear;
close all;
addpath(genpath(fileparts(mfilename('fullpath'))));



startPos = [40, 129, 5];
goalPos  = [951, 833, 10];

[X, Y, Z] = defMap4([1000, 1000, 300]);

params.min_clearance      = 10;
params.max_turn_deg       = 35;
params.max_climb_per_seg  = 10;
params.z_min              = 5;
params.z_max              = 100;
params.n_samples_seg      = 20;
params.big_collision      = 1e12;
params.soft_penalty       = 1e3;

startPos(3) = interp2(X, Y, Z, startPos(1), startPos(2), 'linear', 0) + 3*params.min_clearance;
goalPos(3)  = interp2(X, Y, Z, goalPos(1),  goalPos(2),  'linear', 0) + params.min_clearance;

visualizeTerrain3D(X, Y, Z);
visualizeTerrainContour(X, Y, Z);

popsize   = 200;
nWP       = 5;
dimension = nWP * 3;

xmin = repmat([0, 0, params.z_min], 1, nWP);
xmax = repmat([1000, 1000, params.z_max], 1, nWP);

maxiter = 500;

global UAV_CTX;
UAV_CTX.X = X;
UAV_CTX.params = params;

%% ================== Fitness function ==================
Fitness = @(x) calPathFitnessUAV( ...
    startPos, goalPos, X, Y, Z, decodePath(x), params);

%% ================== Run comparison algorithms ==================
disp('Running SCLDPSO...');
[scldpso_path, scldpso_history] = runSCLDPSO(Fitness, popsize, dimension, xmax, xmin, maxiter);


disp('Running EAPSO...');
[eapso_path, eapso_history] = runEAPSO(Fitness, popsize, dimension, xmax, xmin, maxiter);


disp('Running SLPSO...');
[slpso_path, slpso_history] = runSLPSO(Fitness, popsize, dimension, xmax, xmin, maxiter);

disp('Running CSO...');
[cso_path, cso_history] = runCSO(Fitness, popsize, dimension, xmax, xmin, maxiter);

disp('Running HSLSO...');
[hslso_path, hslso_history] = runHSLSO(Fitness, popsize, dimension, xmax, xmin, maxiter);

%% ================== 3-D path plot ==================
figure;
surf(X, Y, Z, 'EdgeColor', 'none');
hold on;
colormap(turbo);
colorbar;
shading interp;

plotPath(scldpso_path, startPos, goalPos, 'r-', 'SCLDPSO Path');
plotPath(eapso_path,  startPos, goalPos, 'g-', 'EAPSO Path');
plotPath(slpso_path,   startPos, goalPos, 'y-', 'SLPSO Path');
plotPath(cso_path,     startPos, goalPos, 'c-', 'CSO Path');
plotPath(hslso_path,   startPos, goalPos, 'b-', 'HSLSO Path');

scatter3(startPos(1), startPos(2), startPos(3), 100, 'k', 'filled', 'DisplayName', 'Start');
scatter3(goalPos(1),  goalPos(2),  goalPos(3),  100, 'c', 'filled', 'DisplayName', 'Goal');

xlabel('X');
ylabel('Y');
zlabel('Z');
title('UAV 3D Path Planning — Path Comparison');
legend('Location','best');
grid on;
view(-45,35);

%% ================== Top view ==================
figure;
contourf(X, Y, Z, 40, 'LineColor', 'none');
hold on;
colormap(turbo);
colorbar;

plotPath2D(scldpso_path, startPos, goalPos, 'r-', 'SCLDPSO');
plotPath2D(eapso_path,  startPos, goalPos, 'g-', 'EAPSO');
plotPath2D(slpso_path,   startPos, goalPos, 'y-', 'SLPSO');
plotPath2D(cso_path,     startPos, goalPos, 'c-', 'CSO');
plotPath2D(hslso_path,   startPos, goalPos, 'b-', 'HSLSO');

scatter(startPos(1), startPos(2), 80, 'k', 'filled', 'DisplayName', 'Start');
scatter(goalPos(1),  goalPos(2),  80, 'c', 'filled', 'DisplayName', 'Goal');

xlabel('X');
ylabel('Y');
title('Top View of UAV Paths');
axis equal tight;
legend('Location','best');
grid on;

%% ================== Convergence curves ==================
figure;
hold on;

nCurve = 50;
Iters = round(linspace(1, maxiter, nCurve));

plot(Iters, log(max(eapso_history(Iters),  eps)), 'gs-', 'MarkerSize',6, 'LineWidth',1, 'DisplayName','EAPSO');
plot(Iters, log(max(slpso_history(Iters),   eps)), 'yo-', 'MarkerSize',6, 'LineWidth',1, 'DisplayName','SLPSO');
plot(Iters, log(max(cso_history(Iters),     eps)), 'c^-', 'MarkerSize',6, 'LineWidth',1, 'DisplayName','CSO');
plot(Iters, log(max(scldpso_history(Iters), eps)), 'r-<', 'MarkerSize',6, 'LineWidth',1, 'DisplayName','SCLDPSO');
plot(Iters, log(max(hslso_history(Iters),   eps)), 'bv-', 'MarkerSize',6, 'LineWidth',1.5, 'DisplayName','HSLSO');

xlabel('Iterations');
ylabel('log(Best fitness)');
title('Convergence Curves');
legend('Location','best');
grid on;
box on;

%% ================== Metric output ==================
metrics_SCLDPSO = evaluateUAVMetrics(scldpso_path, startPos, goalPos, X, Y, Z, params);
metrics_EAPSO  = evaluateUAVMetrics(eapso_path,  startPos, goalPos, X, Y, Z, params);
metrics_SLPSO   = evaluateUAVMetrics(slpso_path,   startPos, goalPos, X, Y, Z, params);
metrics_CSO     = evaluateUAVMetrics(cso_path,     startPos, goalPos, X, Y, Z, params);
metrics_HSLSO   = evaluateUAVMetrics(hslso_path,   startPos, goalPos, X, Y, Z, params);

fprintf('\n=== Single-run metrics ===\n');
prettyPrintMetrics('SCLDPSO', metrics_SCLDPSO);
prettyPrintMetrics('EAPSO',  metrics_EAPSO);
prettyPrintMetrics('SLPSO',   metrics_SLPSO);
prettyPrintMetrics('CSO',     metrics_CSO);
prettyPrintMetrics('HSLSO',   metrics_HSLSO);
