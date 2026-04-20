%% ========================================================================
%  Project: Rubber Concrete Strength Prediction System (Ablation Study)
%  Author: Li Ding (Hunan University of Technology)
%  Version: Final SCI Integration (Stable & Auto-cleanup Version)
%% ========================================================================
clear; clc; close all;
addpath(genpath(pwd)); 

% --- Module 1: Data Preparation ---
fprintf('>>> Loading datasets...\n');
d2 = readmatrix('数据集2.xlsx'); d2(any(isnan(d2),2),:) = []; 
d3 = readmatrix('数据集3.xlsx'); d3(any(isnan(d3),2),:) = []; 
d4 = readmatrix('数据集4.xlsx'); d4(any(isnan(d4),2),:) = []; 

% --- Module 2: Execute 6 Ablation Experiments ---
fprintf('\n>>> Executing 6 experimental groups (Stable Mode)...\n');
fprintf('------------------------------------------------------------------------------------------------\n');
T_exec = zeros(6, 1); 

% ID 1: Multi-source, Set 2, No PSO
tic; [~, S1, ~] = T3_LSBoost_8_2(d2, false); T_exec(1) = toc;
fprintf('序号 1, 多源异构, 2, 无PSO, 无MaxSize, R2: %.4f, RMSE: %.4f MPa, MAE: %.4f MPa, Time: %.2f s\n', S1.R2_mean, mean(S1.RMSE_test_loop), mean(S1.MAE_test_loop), T_exec(1));

% ID 2: Multi-source, Set 2, Has PSO
tic; [~, S2, ~] = T3_LSBoost_8_2(d2, true);  T_exec(2) = toc;
fprintf('序号 2, 多源异构, 2, 有PSO, 无MaxSize, R2: %.4f, RMSE: %.4f MPa, MAE: %.4f MPa, Time: %.2f s\n', S2.R2_mean, mean(S2.RMSE_test_loop), mean(S2.MAE_test_loop), T_exec(2));

% ID 3: Single, Set 4, No PSO
tic; [~, S3, ~] = T3_LSBoost(d4, false);     T_exec(3) = toc;
fprintf('序号 3, 单源, 4, 无PSO, 无MaxSize, R2: %.4f, RMSE: %.4f MPa, MAE: %.4f MPa, Time: %.2f s\n', S3.R2_mean, mean(S3.RMSE_test_loop), mean(S3.MAE_test_loop), T_exec(3));

% ID 4: Single, Set 3, No PSO
tic; [~, S4, ~] = T3_LSBoost(d3, false);     T_exec(4) = toc;
fprintf('序号 4, 单源, 3, 无PSO, 有MaxSize, R2: %.4f, RMSE: %.4f MPa, MAE: %.4f MPa, Time: %.2f s\n', S4.R2_mean, mean(S4.RMSE_test_loop), mean(S4.MAE_test_loop), T_exec(4));

% ID 5: Single, Set 4, Has PSO
tic; [~, S5, ~] = T3_LSBoost(d4, true);      T_exec(5) = toc;
fprintf('序号 5, 单源, 4, 有PSO, 无MaxSize, R2: %.4f, RMSE: %.4f MPa, MAE: %.4f MPa, Time: %.2f s\n', S5.R2_mean, mean(S5.RMSE_test_loop), mean(S5.MAE_test_loop), T_exec(5));

% ID 6: Single, Set 3, Has PSO
tic; [~, S6, ~] = T3_LSBoost(d3, true);      T_exec(6) = toc;
fprintf('序号 6, 单源, 3, 有PSO, 有MaxSize, R2: %.4f, RMSE: %.4f MPa, MAE: %.4f MPa, Time: %.2f s\n', S6.R2_mean, mean(S6.RMSE_test_loop), mean(S6.MAE_test_loop), T_exec(6));

fprintf('------------------------------------------------------------------------------------------------\n');

% --- Module 3: Statistical Summary ---
Stats = {S1, S2, S3, S4, S5, S6};
R2_V   = cellfun(@(x) x.R2_mean, Stats)';
RMSE_V = cellfun(@(x) mean(x.RMSE_test_loop), Stats)';
MAE_V  = cellfun(@(x) mean(x.MAE_test_loop), Stats)';
Ablation_Table = table((1:6)', {'Multi';'Multi';'Single';'Single';'Single';'Single'}, ...
    {'Set2';'Set2';'Set4';'Set3';'Set4';'Set3'}, {'No';'Yes';'No';'No';'Yes';'Yes'}, ...
    {'No';'No';'No';'Yes';'No';'Yes'}, R2_V, RMSE_V, MAE_V, T_exec, ...
    'VariableNames', {'ID', 'Type', 'Dataset', 'PSO', 'MaxSize', 'R2', 'RMSE_MPa', 'MAE_MPa', 'Time_s'});
format short g
disp('>>> Final Summary Table (English):'); disp(Ablation_Table);

% --- Module 4: Plotting (English) ---
figure('Color', 'w', 'Name', 'SCI_Ablation_Analysis');
subplot(2,1,1); bar(R2_V, 'FaceColor', [0.2 0.4 0.6]); grid on;
ylabel('R^2 Score'); title('(a) Accuracy Comparison'); ylim([min(R2_V)-0.05, 1.0]);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Exp.%d',x), 1:6, 'UniformOutput', false));
subplot(2,1,2); bar([RMSE_V, MAE_V]); legend('RMSE (MPa)', 'MAE (MPa)'); grid on;
ylabel('Error (MPa)'); title('(b) Error Comparison');
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Exp.%d',x), 1:6, 'UniformOutput', false));

exportgraphics(figure(1), 'Performance_Results.png', 'Resolution', 300);
save('Ablation_Full_Results.mat', 'Ablation_Table');
fprintf('\n✅ Completed! Results saved as PNG.\n');