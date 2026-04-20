%% ========================================================================
%  Project: Rubber Concrete Strength Prediction System (Ablation Study)
%  Author: Li Ding (Hunan University of Technology)
%  Version: Final Ultimate Version (SCI Standards & Sound Alert)
%% ========================================================================
clear; clc; close all;
addpath(genpath(pwd)); 

% --- Module 1: Data Preparation ---
fprintf('>>> Loading datasets...\n');
d2 = readmatrix('数据集2.xlsx'); d2(any(isnan(d2),2),:) = []; 
d3 = readmatrix('数据集3.xlsx'); d3(any(isnan(d3),2),:) = []; 
d4 = readmatrix('数据集4.xlsx'); d4(any(isnan(d4),2),:) = []; 

% --- Module 2: Breakpoint Recovery (断点记录系统) ---
backup_file = 'Ablation_Progress_Backup.mat';
if exist(backup_file, 'file')
    fprintf('>>> 检测到未完成的备份，正在恢复进度...\n');
    load(backup_file); 
else
    Stats_All = cell(6, 1); T_exec = zeros(6, 1);
end

% --- Module 3: Execution Loop with Auto-Save (稳健执行与实时存盘) ---
fprintf('\n>>> 开始执行消融实验任务清单...\n');

% [1] 多源异构 | 2 | 无PSO | 无MaxSize
if isempty(Stats_All{1})
    tic; [~, S1, ~] = T3_LSBoost_8_2(d2, false); T_exec(1) = toc;
    Stats_All{1} = S1; save(backup_file, 'Stats_All', 'T_exec');
    fprintf('序号 1, 多源异构, 数据集2, 无PSO, 无MaxSize, R2: %.4f, RMSE: %.4f, MAE: %.4f, Time: %.2f s\n', S1.R2_mean, mean(S1.RMSE_test_loop), mean(S1.MAE_test_loop), T_exec(1));
end

% [2] 多源异构 | 2 | 有PSO | 无MaxSize
if isempty(Stats_All{2})
    fprintf('Running Exp.2 (PSO Optimized, please wait)...\n');
    tic; [~, S2, ~] = T3_LSBoost_8_2(d2, true); T_exec(2) = toc;
    Stats_All{2} = S2; save(backup_file, 'Stats_All', 'T_exec');
    fprintf('序号 2, 多源异构, 数据集2, 有PSO, 无MaxSize, R2: %.4f, RMSE: %.4f, MAE: %.4f, Time: %.2f s\n', S2.R2_mean, mean(S2.RMSE_test_loop), mean(S2.MAE_test_loop), T_exec(2));
    java.lang.System.gc(); pause(5); % 内存深度清理
end

% [3] 单源 | 4 | 无PSO | 无MaxSize
if isempty(Stats_All{3})
    tic; [~, S3, ~] = T3_LSBoost(d4, false); T_exec(3) = toc;
    Stats_All{3} = S3; save(backup_file, 'Stats_All', 'T_exec');
    fprintf('序号 3, 单源, 数据集4, 无PSO, 无MaxSize, R2: %.4f, RMSE: %.4f, MAE: %.4f, Time: %.2f s\n', S3.R2_mean, mean(S3.RMSE_test_loop), mean(S3.MAE_test_loop), T_exec(3));
end

% [4] 单源 | 3 | 无PSO | 有MaxSize
if isempty(Stats_All{4})
    tic; [~, S4, ~] = T3_LSBoost(d3, false); T_exec(4) = toc;
    Stats_All{4} = S4; save(backup_file, 'Stats_All', 'T_exec');
    fprintf('序号 4, 单源, 数据集3, 无PSO, 有MaxSize, R2: %.4f, RMSE: %.4f, MAE: %.4f, Time: %.2f s\n', S4.R2_mean, mean(S4.RMSE_test_loop), mean(S4.MAE_test_loop), T_exec(4));
end

% [5] 单源 | 4 | 有PSO | 无MaxSize
if isempty(Stats_All{5})
    fprintf('Running Exp.5 (PSO Optimized, please wait)...\n');
    tic; [~, S5, ~] = T3_LSBoost(d4, true); T_exec(5) = toc;
    Stats_All{5} = S5; save(backup_file, 'Stats_All', 'T_exec');
    fprintf('序号 5, 单源, 数据集4, 有PSO, 无MaxSize, R2: %.4f, RMSE: %.4f, MAE: %.4f, Time: %.2f s\n', S5.R2_mean, mean(S5.RMSE_test_loop), mean(S5.MAE_test_loop), T_exec(5));
    java.lang.System.gc(); pause(5);
end

% [6] 单源 | 3 | 有PSO | 有MaxSize
if isempty(Stats_All{6})
    fprintf('Running Exp.6 (PSO Optimized, please wait)...\n');
    tic; [~, S6, ~] = T3_LSBoost(d3, true); T_exec(6) = toc;
    Stats_All{6} = S6; save(backup_file, 'Stats_All', 'T_exec');
    fprintf('序号 6, 单源, 数据集3, 有PSO, 有MaxSize, R2: %.4f, RMSE: %.4f, MAE: %.4f, Time: %.2f s\n', S6.R2_mean, mean(S6.RMSE_test_loop), mean(S6.MAE_test_loop), T_exec(6));
end

% --- Module 4: Statistics Summary ---
R2_V = cellfun(@(x) x.R2_mean, Stats_All);
RMSE_V = cellfun(@(x) mean(x.RMSE_test_loop), Stats_All);
MAE_V = cellfun(@(x) mean(x.MAE_test_loop), Stats_All);

% --- Module 5: Visualization (English Labels + Under-axis Titles + Trend Lines) ---
fig = figure('Color', 'w', 'Name', 'SCI_Performance_Analysis', 'Position', [50, 50, 1000, 980]);
exp_x = 1:6;

% (a) Accuracy R^2
subplot(4,1,1);
bar(exp_x, R2_V, 0.5, 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'k'); hold on;
plot(exp_x, R2_V, '-o', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
text(exp_x, R2_V, string(num2str(R2_V, '%.4f')), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
ylabel('R^2 Score'); ylim([min(R2_V)-0.05, 1.05]); grid on;
title('(a) Global Accuracy Comparison', 'VerticalAlignment', 'top', 'Units', 'normalized', 'Position', [0.5, -0.3, 0]);

% (b) RMSE
subplot(4,1,2);
bar(exp_x, RMSE_V, 0.5, 'FaceColor', [0.2 0.6 0.4], 'EdgeColor', 'k'); hold on;
plot(exp_x, RMSE_V, '-s', 'Color', [0.1 0.3 0.1], 'LineWidth', 1.2, 'MarkerFaceColor', 'w');
text(exp_x, RMSE_V, string(num2str(RMSE_V, '%.2f')), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
ylabel('RMSE (MPa)'); grid on;
title('(b) Global RMSE Comparison', 'VerticalAlignment', 'top', 'Units', 'normalized', 'Position', [0.5, -0.3, 0]);

% (c) MAE
subplot(4,1,3);
bar(exp_x, MAE_V, 0.5, 'FaceColor', [0.7 0.5 0.2], 'EdgeColor', 'k'); hold on;
plot(exp_x, MAE_V, '-d', 'Color', [0.4 0.2 0.1], 'LineWidth', 1.2, 'MarkerFaceColor', 'w');
text(exp_x, MAE_V, string(num2str(MAE_V, '%.2f')), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
ylabel('MAE (MPa)'); grid on;
title('(c) Global MAE Comparison', 'VerticalAlignment', 'top', 'Units', 'normalized', 'Position', [0.5, -0.3, 0]);

% (d) Time
subplot(4,1,4);
bar(exp_x, T_exec, 0.5, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k'); hold on;
plot(exp_x, T_exec, '-^', 'Color', 'k', 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
text(exp_x, T_exec, string(num2str(T_exec, '%.2f')), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
ylabel('Time (s)'); grid on;
title('(d) Runtime Comparison', 'VerticalAlignment', 'top', 'Units', 'normalized', 'Position', [0.5, -0.3, 0]);

% Formatting X-axis
for k = 1:4
    subplot(4,1,k); set(gca, 'XTick', 1:6, 'XTickLabel', arrayfun(@(x) sprintf('Exp.%d',x), 1:6, 'UniformOutput', false));
end

exportgraphics(fig, 'Final_Ablation_Analysis_Report.png', 'Resolution', 300);
delete(backup_file); % 任务完成后删除备份

% --- Module 6: Completion Notification (Sound Alert) ---
fprintf('\n🔔 所有任务圆满完成！\n');
for beep_i = 1:3
    beep; pause(0.8); % 连响三次温和提示音
end
msgbox('Ablation Study Sequence Finished Successfully!', 'Success', 'help');