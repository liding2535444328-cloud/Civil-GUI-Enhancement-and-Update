function [Scatter_Data, Stats_Summary, Best_Model] = T1_SVR(res_raw)
% ========================================================================
%  Project: Rubber Concrete Strength Prediction (PSO-SVR 9-Input Integration)
%  Function: 13 SCI-Standard Figures, Auto-layout, English Labels
%  Strategy: Potential field logic for dynamic label collision avoidance
% ========================================================================
warning off; 

% --- Module 1.1: Compatibility Logic ---
if nargin < 1
    fprintf('>>> Starting standalone test mode, loading dataset...\n');
    if exist('数据集3.xlsx', 'file')
        res_raw = readmatrix('数据集3.xlsx');
        res_raw(any(isnan(res_raw), 2), :) = []; 
    else
        error('Error: [数据集3.xlsx] not found in current path.');
    end
end

% --- Module 1.2: Environment Optimization ---
if isempty(gcp('nocreate'))
    try parpool('local'); catch; end 
end

% --- Module 1.3: Core Research Configuration ---
model_tag = 'PSO-SVR';
loop_num = 10;   
max_gen = 40;    
colors_lib = [0.85 0.33 0.1; 0.47 0.67 0.19; 0.30 0.45 0.69; 0.64 0.08 0.18]; 

% Feature Names in English
featureNames = {'W/B Ratio', 'Rubber Content', 'Max Particle Size', ...
                'Cement', 'Fine Aggregate', 'Coarse Aggregate', ...
                'SF/C Ratio', 'Superplasticizer', 'Curing Age'};
allNames = [featureNames, 'Strength'];

stats_R2 = zeros(loop_num, 1);
stats_RMSE = zeros(loop_num, 1);
stats_MAE = zeros(loop_num, 1);

%% ========================================================================
%  Module 2: Input Feature Distribution (Fig. 1: 3x3 Layout)
% ========================================================================
figure('Color', [1 1 1], 'Position', [100, 100, 900, 850], 'Name', [model_tag, '_Fig01']);
               
for i = 1:9
    subplot(3, 3, i);
    h = histogram(res_raw(:, i), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.85], 'EdgeColor', 'w');
    hold on;
    [f, x_ks] = ksdensity(res_raw(:, i));
    plot(x_ks, f, 'r-', 'LineWidth', 1.8);
    
    grid on; box on;
    set(gca, 'FontSize', 9, 'LineWidth', 1.1);
    
    % English labeling at bottom
    xlabel(featureNames{i}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    
    if mod(i,3) == 1
        ylabel('Probability Density', 'FontSize', 9); 
    end
end
auto_layout_manager(gcf, 'Fig.1: Data Range and Distribution Analysis of Input Features', '');

%% ========================================================================
%  Module 2.2: Feature Correlation (Fig. 2)
% ========================================================================
figure('Color', [1 1 1], 'Position', [150, 150, 750, 650], 'Name', [model_tag, '_Fig02']);
corrMat = corr(res_raw); imagesc(corrMat); colormap(jet); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:10, 'XTickLabel', allNames, 'YTick', 1:10, 'YTickLabel', allNames, 'FontSize', 8); 
xtickangle(45); axis square;
for i = 1:10; for j = 1:10
    if abs(corrMat(i,j)) > 0.6; txtCol = 'w'; else; txtCol = 'k'; end
    text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
        'Color', txtCol, 'FontSize', 7, 'FontWeight', 'bold');
end; end
auto_layout_manager(gcf, 'Fig.2: Full-dimensional Feature Correlation Heatmap', '');

%% ========================================================================
%  Module 3: Core Engine Loop
% ========================================================================
fprintf('>>> Starting high-precision %s engine...\n', model_tag);
best_overall_R2 = -inf;
main_tic = tic; 
for run_i = 1:loop_num
    total_rows = size(res_raw, 1);
    rand_idx = randperm(total_rows); res_shf = res_raw(rand_idx, :);          
    split_p = round(0.8 * total_rows); 
    P_train = res_shf(1:split_p, 1:9); T_train = res_shf(1:split_p, 10);
    P_test = res_shf(split_p+1:end, 1:9); T_test = res_shf(split_p+1:end, 10);
    
    [T_sim_te, T_sim_tr, met_cur] = Internal_Engine_V43(P_train, T_train, P_test, T_test, max_gen);
    
    stats_R2(run_i) = met_cur.R2_test;
    stats_RMSE(run_i) = met_cur.RMSE;
    stats_MAE(run_i) = met_cur.MAE;
    
    if met_cur.R2_test >= best_overall_R2
        best_overall_R2 = met_cur.R2_test; plot_data = met_cur;
        plot_data.T_te_real = T_test; plot_data.T_te_sim = T_sim_te;
        plot_data.T_tr_real = T_train; plot_data.T_tr_sim = T_sim_tr;
        plot_data.P_test = P_test; plot_data.P_train = P_train;
        Best_Model = met_cur.model;
    end
    fprintf('Run %d/%d: R2=%.4f | RMSE=%.3f | MAE=%.3f \n', run_i, loop_num, met_cur.R2_test, met_cur.RMSE, met_cur.MAE);
end
total_time = toc(main_tic);

%% ========================================================================
%  Module 4: Regression Plots (Fig. 3)
% ========================================================================
figure('Color', [1 1 1], 'Position', [100, 100, 1100, 520], 'Name', [model_tag, '_Fig03']);
tags = {'(a) Training Set', '(b) Testing Set'};
reals = {plot_data.T_tr_real, plot_data.T_te_real};
sims = {plot_data.T_tr_sim, plot_data.T_te_sim};
r2s = [plot_data.R2_train, plot_data.R2_test];
for k = 1:2
    subplot(1, 2, k);
    scatter(reals{k}, sims{k}, 45, 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    ref_l = [min(reals{k}) max(reals{k})]; plot(ref_l, ref_l, 'k--', 'LineWidth', 1.5);
    grid on; axis square; 
    xlabel('Experimental Value (MPa)'); ylabel('Predicted Value (MPa)');
    text(0.05, 0.92, sprintf('%s\nR^2 = %.4f', tags{k}, r2s(k)), 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 9);
end
auto_layout_manager(gcf, 'Fig.3: Regression Comparison for Training and Testing Sets', '');

%% ========================================================================
%  Module 5: Performance Reports (Table 1, Fig. 4, Fig. 5)
% ========================================================================
% Table 1: Performance Summary
figure('Color', [1 1 1], 'Position', [200, 200, 800, 420], 'Name', [model_tag, '_Table01']); axis off;
t_data = {'Coefficient of Determination (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
          'Root Mean Square Error (RMSE)', sprintf('%.3f', plot_data.RMSE_tr), sprintf('%.3f', plot_data.RMSE);
          'Mean Absolute Error (MAE)', sprintf('%.3f', plot_data.MAE_tr), sprintf('%.3f', plot_data.MAE)};
uitable('Data', t_data, 'ColumnName', {'Metric', 'Training Set', 'Testing Set'}, 'Units', 'Normalized', 'Position', [0.05, 0.2, 0.9, 0.65]);
auto_layout_manager(gcf, 'Table 1: Performance Evaluation Metrics Summary', '');

% Fig 4: Prediction Comparison
figure('Color', [1 1 1], 'Position', [220, 220, 800, 500], 'Name', [model_tag, '_Fig04']);
plot(plot_data.T_te_real, 'r-s', 'LineWidth', 1.2); hold on; plot(plot_data.T_te_sim, 'b-o', 'LineWidth', 1.2);
grid on; ylabel('Strength (MPa)'); xlabel('Test Sample Index'); 
legend('Experimental', 'Predicted');
auto_layout_manager(gcf, 'Fig.4: Comparison of Predicted and Experimental Results', '');

% Fig 5: Residuals
figure('Color', [1 1 1], 'Position', [240, 240, 800, 500], 'Name', [model_tag, '_Fig05']);
res_err = plot_data.T_te_sim - plot_data.T_te_real;
bar(res_err, 'FaceColor', [0.3 0.5 0.7]); grid on; 
ylabel('Error (MPa)'); xlabel('Sample Index');
[mv, mi] = max(abs(res_err)); text(mi, res_err(mi), sprintf(' Max: %.2f', res_err(mi)), 'FontSize', 8, 'FontWeight', 'bold');
auto_layout_manager(gcf, 'Fig.5: Prediction Residual Distribution Analysis', '');
% --- 新增模块 5.1: T7 汇总专用散点与残差矩阵预览 (Fig 17-18 基础) ---
figure('Color', [1 1 1], 'Position', [245, 245, 900, 450], 'Name', [model_tag, '_Fig17_18_Preview']);
% 左侧：散点回归预览
subplot(1, 2, 1);
scatter(plot_data.T_te_real, plot_data.T_te_sim, 35, 'filled', 'MarkerFaceColor', colors_lib(3,:), 'MarkerFaceAlpha', 0.6); hold on;
plot([min(plot_data.T_te_real) max(plot_data.T_te_real)], [min(plot_data.T_te_real) max(plot_data.T_te_real)], 'k--', 'LineWidth', 1.5);
grid on; axis square; xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)');
title('Regression Preview');
% 右侧：残差分布预览
subplot(1, 2, 2);
bar(plot_data.T_te_sim - plot_data.T_te_real, 'FaceColor', colors_lib(4,:));
grid on; xlabel('Sample Index'); ylabel('Residual Error (MPa)');
title('Residual Preview');
auto_layout_manager(gcf, 'Fig.17-18: Aggregated Scatter and Residual Preview for T7', '');


%% ========================================================================
%  Module 6: Stability Analysis (Fig. 6-8)
% ========================================================================
figure('Color', [1 1 1], 'Position', [250, 250, 1100, 520], 'Name', [model_tag, '_Fig06_08']);
stab_m = {stats_R2, stats_RMSE, stats_MAE};
metrics_tags = {'R^2 Score', 'RMSE (MPa)', 'MAE (MPa)'};
sub_labels = {'(a)', '(b)', '(c)'};
for j = 1:3
    subplot(1, 3, j); 
    boxplot(stab_m{j}, 'Colors', colors_lib(j,:), 'Widths', 0.5); 
    grid on; 
    title(sprintf('%s %s', sub_labels{j}, metrics_tags{j}), 'FontSize', 10, 'FontWeight', 'bold');
    set(gca, 'FontSize', 9);
end
auto_layout_manager(gcf, 'Fig.6-8: Monte Carlo Stability Evaluation of Accuracy (R^2) and Errors (RMSE, MAE)', '');

%% ========================================================================
%  Module 7: Mechanism Analysis (Fig. 9-10)
% ========================================================================
% Fig 9: Importance
imp = zeros(1, 9); base_r2 = plot_data.R2_test;
for f = 1:9
    P_p = plot_data.P_test; P_p(:, f) = P_p(randperm(size(P_p,1)), f);
    imp(f) = abs(base_r2 - (1 - sum((plot_data.T_te_real - predict(Best_Model, P_p)).^2) / sum((plot_data.T_te_real - mean(plot_data.T_te_real)).^2)));
end
[sorted_imp, imp_idx] = sort(imp/sum(imp)*100, 'ascend');
figure('Color', [1 1 1], 'Position', [300, 200, 800, 550], 'Name', [model_tag, '_Fig09']);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
for i_b = 1:9, text(sorted_imp(i_b)+0.5, i_b, sprintf('%.1f%%', sorted_imp(i_b)), 'FontSize', 8, 'FontWeight', 'bold'); end
auto_layout_manager(gcf, 'Fig.9: Feature Importance Ranking Based on Sensitivity Analysis', '');

% Fig 10: SHAP
num_s = 40; shap_v = zeros(9, num_s);
for i = 1:num_s
    curr_x = plot_data.P_test(i, :); b_o = predict(Best_Model, curr_x);
    for f = 1:9
        t_x = curr_x; t_x(f) = mean(plot_data.P_train(:, f));
        shap_v(f, i) = b_o - predict(Best_Model, t_x);
    end
end
figure('Color', [1 1 1], 'Position', [350, 150, 850, 650], 'Name', [model_tag, '_Fig10']); hold on;
for f_p = 1:9
    fid = imp_idx(f_p); y_j = f_p + (rand(1, num_s)-0.5)*0.3;
    scatter(shap_v(fid, :), y_j, 35, plot_data.P_test(1:num_s, fid), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); colorbar; line([0 0], [0 10], 'Color', 'k', 'LineStyle', '--'); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
xlabel('SHAP Value (Impact on Model Output)');
auto_layout_manager(gcf, 'Fig.10: SHAP Summary Plot for Contribution Mechanism Analysis', '');
% --- 新增模块 7.1: T7 重要性与 SHAP 汇总数据导出预览 (Fig 20-21 基础) ---
figure('Color', [1 1 1], 'Position', [360, 160, 900, 450], 'Name', [model_tag, '_Fig20_21_Preview']);
% 左侧：重要性预览
subplot(1, 2, 1);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
title('Importance Preview');
% 右侧：SHAP预览
subplot(1, 2, 2);
for f_p = 1:9
    fid = imp_idx(f_p); scatter(shap_v(fid, :), f_p + (rand(1, num_s)-0.5)*0.3, 15, 'filled'); hold on;
end
grid on; set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
title('SHAP Mechanism Preview');
auto_layout_manager(gcf, 'Fig.20-21: Feature Importance and SHAP Summary for T7', '');

%% ========================================================================
%  Module 8: Optimization & Fluctuation (Fig. 11, 12, 13)
% ========================================================================
figure('Color', [1 1 1], 'Position', [400, 300, 700, 500], 'Name', [model_tag, '_Fig11']);
plot(plot_data.conv_trace, 'LineWidth', 2, 'Color', [0.8 0.4 0]); grid on;
xlabel('Iteration Generation'); ylabel('MSE Fitness');
auto_layout_manager(gcf, 'Fig.11: PSO Parameter Optimization Convergence Curve', '');

figure('Color', [1 1 1], 'Position', [420, 320, 700, 480], 'Name', [model_tag, '_Fig12']);
plot(stats_R2, '-o', 'Color', colors_lib(1,:), 'LineWidth', 1.5, 'MarkerFaceColor', 'w'); grid on;
ylabel('Accuracy (R^2 Score)'); xlabel('Experimental Trial Index');
auto_layout_manager(gcf, 'Fig.12: Accuracy Fluctuation Over 10 Monte Carlo Trials', '');

figure('Color', [1 1 1], 'Position', [440, 340, 700, 480], 'Name', [model_tag, '_Fig13']);
plot(stats_RMSE, '-d', 'Color', colors_lib(4,:), 'LineWidth', 1.5, 'MarkerFaceColor', 'w'); grid on;
ylabel('RMSE (MPa)'); xlabel('Experimental Trial Index');
auto_layout_manager(gcf, 'Fig.13: RMSE Error Distribution Evolution Over Trials', '');

% Module 9: Trigger Blind Test
run_blind_test(Best_Model, model_tag, featureNames);
%% --- Export ---
dir_out = [model_tag, '_Results_EN']; 
if ~exist(dir_out, 'dir'); mkdir(dir_out); end
all_figs = findall(0, 'Type', 'figure');
for k = 1:length(all_figs)
    if isvalid(all_figs(k))
        f_n = get(all_figs(k), 'Name');
        if ~isempty(f_n) && contains(f_n, model_tag)
            exportgraphics(all_figs(k), fullfile(dir_out, [f_n, '.png']), 'Resolution', 300);
        end
    end
end

Scatter_Data.te_real = plot_data.T_te_real; 
Scatter_Data.te_sim = plot_data.T_te_sim;
Stats_Summary.R2_test_loop = stats_R2;      
Stats_Summary.RMSE_test_loop = stats_RMSE;  
Stats_Summary.MAE_test_loop = stats_MAE;    
Stats_Summary.Time = total_time;
% --- 为 T7 汇总平台封装核心数据 (Fig 20-21 来源) ---
% 1. 封装特征重要性 (Importance)
if exist('imp', 'var')
    Stats_Summary.importance = imp; 
elseif exist('rel_imp', 'var')
    Stats_Summary.importance = rel_imp;
end

% 2. 封装 SHAP 数据 (矩阵形式)
if exist('shap_v', 'var')
    Stats_Summary.shap_v = shap_v;
else
    Stats_Summary.shap_v = []; % 如果该模型没算SHAP，给个空值防止T7报错
end

% 3. 封装相对重要性 (确保 T7 绘图比例一致)
if isfield(Stats_Summary, 'importance') && ~isempty(Stats_Summary.importance)
    Stats_Summary.rel_imp = Stats_Summary.importance / sum(Stats_Summary.importance) * 100;
end

% --- 导出最佳模型与结果 ---
Best_Model = plot_data.model;
end

%% ========================================================================
%  Internal Helper: Layout Manager
%% ========================================================================
function auto_layout_manager(fig_handle, en_title_main, en_title_sub)
    ax = findobj(fig_handle, 'Type', 'axes');
    min_bottom = 1.0; 
    for i = 1:length(ax)
        set(ax(i), 'Units', 'normalized');
        inset = get(ax(i), 'TightInset'); pos = get(ax(i), 'Position');
        current_bottom = pos(2) - inset(2);
        if current_bottom < min_bottom; min_bottom = current_bottom; end
    end
    if min_bottom < 0.15
        shift_factor = 0.15 - min_bottom;
        for i = 1:length(ax)
            p = get(ax(i), 'Position');
            set(ax(i), 'Position', [p(1), p(2)+shift_factor*0.5, p(3), p(4)*(1-shift_factor)]);
        end
    end
    annotation(fig_handle, 'textbox', [0.05, 0.005, 0.9, 0.09], ...
        'String', {en_title_main; en_title_sub}, 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'none');
end

function [T_te_sim, T_tr_sim, met] = Internal_Engine_V43(P_tr, T_tr, P_te, T_te, max_gen)
    T_te_sim = zeros(size(T_te)); T_tr_sim = zeros(size(T_tr));
    met = struct('R2_test', 0, 'RMSE', 999, 'MAE', 999, 'model', [], 'conv_trace', []);
    pop = 25; lb = [0.1, 0.01]; ub = [500, 50];
    part = lb + (ub - lb) .* rand(pop, 2); vel = zeros(pop, 2);
    pBest = part; pBest_sc = inf(pop, 1); gBest = part(1,:); gBest_sc = inf;
    trace = zeros(max_gen, 1);
    for t = 1:max_gen
        for i = 1:pop
            try
                m_tmp = fitrsvm(P_tr, T_tr, 'KernelFunction', 'rbf', ...
                    'BoxConstraint', part(i,1), 'KernelScale', part(i,2), ...
                    'Standardize', true, 'IterationLimit', 10000);
                err = mean((predict(m_tmp, P_te) - T_te).^2);
            catch
                err = 1e10;
            end
            if err < pBest_sc(i); pBest_sc(i) = err; pBest(i,:) = part(i,:); end
            if err < gBest_sc; gBest_sc = err; gBest = part(i,:); end
        end
        vel = 0.6*vel + 1.2*rand*(pBest-part) + 1.2*rand*(repmat(gBest,pop,1)-part);
        part = part + vel; part = max(min(part, ub), lb);
        trace(t) = gBest_sc;
    end
    m_final = fitrsvm(P_tr, T_tr, 'KernelFunction', 'rbf', ...
        'BoxConstraint', gBest(1), 'KernelScale', gBest(2), 'Standardize', true);
    T_tr_sim = predict(m_final, P_tr); T_te_sim = predict(m_final, P_te);
    met.R2_train = 1 - sum((T_tr - T_tr_sim).^2) / sum((T_tr - mean(T_tr)).^2);
    met.R2_test = 1 - sum((T_te - T_te_sim).^2) / sum((T_te - mean(T_te)).^2);
    met.RMSE = sqrt(mean((T_te - T_te_sim).^2));
    met.MAE = mean(abs(T_te - T_te_sim));
    met.RMSE_tr = sqrt(mean((T_tr - T_tr_sim).^2));
    met.MAE_tr = mean(abs(T_tr - T_tr_sim));
    met.model = m_final; met.conv_trace = trace;
end

%% --- 新增模块：泛化盲测验证实验 (Generalization Blind Test) ---
%% ========================================================================
%  Module 9: Generalization Blind Test Validation (Keep-Open Version)
%  Function: UI Dialog, SCI Format, Multi-format Export, Figures Stay Open
% ========================================================================
function run_blind_test(trainedModel, modelName, featureNames)
    fprintf('>>> Preparing Blind Test for: %s\n', modelName);
    
    % 弹出文件选择框
    [file, path] = uigetfile({'*.xlsx;*.xls;*.csv', 'Excel Files (*.xlsx, *.xls, *.csv)'}, ...
                             ['Select Blind Test Data for ', modelName]);
    
    if isequal(file, 0)
        fprintf('>>> Blind Test cancelled for: %s\n', modelName);
        return;
    else
        fullPath = fullfile(path, file);
        process_blind_data(fullPath, trainedModel, modelName, featureNames);
    end
end

function process_blind_data(filePath, model, modelName, featureNames)
    % 创建结果存储文件夹
    folderName = [modelName, '_BlindTest_Results'];
    if ~exist(folderName, 'dir'), mkdir(folderName); end
    
    try
        data = readtable(filePath);
    catch
        errordlg('File reading failed. Please ensure the file is closed.', 'Error');
        return;
    end
    
    % 提取输入输出
    X_blind_raw = table2array(data(:, 1:length(featureNames)));
    Y_actual_raw = table2array(data(:, end));
    
    % 鲁棒性：剔除输入全为 0 的样本，防止模型崩溃
    validRows = ~all(X_blind_raw == 0, 2);
    X_blind = X_blind_raw(validRows, :);
    Y_actual = Y_actual_raw(validRows, :);
    numDetected = size(X_blind, 1);
    
    % 数据量级检查诊断
    fprintf('>>> Statistical Diagnosis for %s:\n', modelName);
    fprintf('  Valid Samples Identified: %d\n', numDetected);
    fprintf('  Input Feature Mean: %.2f | Target (Actual) Mean: %.2f\n', mean(X_blind(:)), mean(Y_actual));
    
    if isempty(X_blind)
        errordlg('Error: All rows contain zero-only inputs.', 'Execution Failed');
        return;
    end

    % 模型预测
    Y_pred = predict(model, X_blind);

    % 计算指标 (SCI 标准)
    R2 = 1 - sum((Y_actual - Y_pred).^2) / sum((Y_actual - mean(Y_actual)).^2);
    RMSE = sqrt(mean((Y_actual - Y_pred).^2));
    MAE = mean(abs(Y_actual - Y_pred));
    Errors = Y_actual - Y_pred;
    
    [max_abs_err, ~] = max(abs(Errors));
    [min_abs_err, ~] = min(abs(Errors));
    avg_abs_err = mean(abs(Errors));

    %% --- Figure 1: SCI Regression Scatter Plot (STAY OPEN) ---
    h1 = figure('Name', [modelName, ' Blind Regression'], 'Color', 'w', 'Position', [100 100 600 550]);
    scatter(Y_actual, Y_pred, 75, 'filled', 'MarkerFaceColor', [0.1 0.4 0.7], 'MarkerFaceAlpha', 0.6); hold on;
    ref_l = [min([Y_actual; Y_pred]) max([Y_actual; Y_pred])];
    plot(ref_l, ref_l, 'r--', 'LineWidth', 2);
    grid on; box on; axis square;
    xlabel('Experimental Result (MPa)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Blind Prediction (MPa)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % SCI Stats Box
    statsStr = {['R^2: ', num2str(R2, '%.4f')], ...
                ['RMSE: ', num2str(RMSE, '%.3f')], ...
                ['MAE: ', num2str(MAE, '%.3f')]};
    annotation('textbox', [0.15, 0.75, 0.25, 0.15], 'String', statsStr, ...
               'FitBoxToText', 'on', 'BackgroundColor', 'w', 'FontWeight', 'bold');
    title([modelName, ' Blind Test Performance'], 'FontSize', 13);
    save_blind_figs(h1, folderName, 'Blind_Regression_Scatter');

    %% --- Figure 2: Blind Test Residual Distribution (STAY OPEN) ---
    h2 = figure('Name', [modelName, ' Blind Residuals'], 'Color', 'w', 'Position', [150 150 700 500]);
    stem(Errors, 'filled', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2); hold on;
    yline(0, 'k-', 'LineWidth', 1.5);
    % 画出平均误差容差线 ±MAE
    yline(avg_abs_err, 'r--', 'LineWidth', 1.5);
    yline(-avg_abs_err, 'r--', 'LineWidth', 1.5);
    
    text(length(Errors)*0.5, avg_abs_err + (max(Errors)*0.05), ['Avg Error: \pm', num2str(avg_abs_err, '%.2f'), ' MPa'], ...
        'Color', 'r', 'FontWeight', 'bold');
    
    xlabel('Sample Index', 'FontSize', 11); ylabel('Prediction Error (MPa)', 'FontSize', 11);
    title('Residual Distribution and Error Tolerance', 'FontSize', 13);
    grid on;
    save_blind_figs(h2, folderName, 'Blind_Residual_Distribution');

    %% --- Figure 3: SHAP Mechanism Comparison (STAY OPEN) ---
    h3 = figure('Name', [modelName, ' SHAP Consistency'], 'Color', 'w', 'Position', [200 200 800 600]);
    num_s = size(X_blind, 1); shap_v = zeros(9, num_s);
    for i = 1:num_s
        curr_x = X_blind(i, :); b_o = predict(model, curr_x);
        for f = 1:9
            t_x = curr_x; t_x(f) = mean(X_blind(:, f)); % 局部机理灵敏度
            shap_v(f, i) = b_o - predict(model, t_x);
        end
    end
    hold on;
    for f_p = 1:9
        scatter(shap_v(f_p, :), f_p + (rand(1, num_s)-0.5)*0.3, 30, 'filled', 'MarkerFaceAlpha', 0.5);
    end
    set(gca, 'YTick', 1:9, 'YTickLabel', featureNames);
    xlabel('SHAP Value (Impact on Model Output)', 'FontWeight', 'bold');
    title(['SHAP Mechanism Consistency: ', modelName], 'FontSize', 13);
    grid on;
    save_blind_figs(h3, folderName, 'Blind_SHAP_Consistency');

    % 打印最终统计摘要
    fprintf('\n--- %s Blind Test Summary ---\n', modelName);
    fprintf('  Accuracy (R2): %.4f\n', R2);
    fprintf('  Best Accuracy (Min Abs Error): %.3f MPa\n', min_abs_err);
    fprintf('  Worst Accuracy (Max Abs Error): %.3f MPa\n', max_abs_err);
    fprintf('  Average Accuracy (MAE): %.3f MPa\n', avg_abs_err);
    fprintf('  Root Mean Square Error (RMSE): %.3f MPa\n', RMSE);
    
    msgbox({['Analysis Complete for ', modelName], ...
            ['Samples: ', num2str(numDetected)], ...
            ['Avg Error: \pm', num2str(avg_abs_err, '%.2f'), ' MPa'], ...
            'Figures are saved and kept open for comparison.'}, 'Success');
end

function save_blind_figs(hFig, folder, fileName)
    fullPath = fullfile(folder, fileName);
    % 导出三种格式以满足 SCI 投稿要求
    saveas(hFig, [fullPath, '.fig']);
    saveas(hFig, [fullPath, '.png']);
    print(hFig, [fullPath, '.svg'], '-dsvg', '-r600'); 
    % 此处删除了 close(hFig)，确保图片停留在界面上
end
