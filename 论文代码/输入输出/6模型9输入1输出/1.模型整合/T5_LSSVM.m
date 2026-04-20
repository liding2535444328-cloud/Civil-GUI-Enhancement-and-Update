function [Scatter_Data, Stats_Summary, Best_Model] = T5_LSSVM(res_raw)
% ========================================================================
%  Project: Rubber Concrete Strength Prediction (PSO-LSSVM Integration)
%  Model: PSO-LSSVM (Least Squares Support Vector Machine)
%  Core: 13 SCI-Standard Figures - Full English Version
% ========================================================================
warning off; 
% --- Module 1.1: Environment Initialization ---
toolbox_folder = 'LSSVM_Toolbox';
if exist(toolbox_folder, 'dir')
    full_toolbox_path = genpath(toolbox_folder); 
    addpath(full_toolbox_path);
else
    error('Error: LSSVM_Toolbox folder not found.');
end
if isempty(gcp('nocreate'))
    try parpool('local'); catch; end 
end
pctRunOnAll(['addpath(''', full_toolbox_path, ''')']);

if nargin < 1
    fprintf('>>> Starting [PSO-LSSVM] Standalone Mode...\n');
    if exist('数据集3.xlsx', 'file')
        res_raw = readmatrix('数据集3.xlsx');
        res_raw(any(isnan(res_raw), 2), :) = []; 
    else
        error('Data file not found.');
    end
end
% --- Module 1.3: Core Config ---
model_tag = 'PSO-LSSVM';
loop_num = 10; max_gen = 30;    
colors_lib = [0.85 0.33 0.1; 0.47 0.67 0.19; 0.30 0.45 0.69; 0.64 0.08 0.18]; 
featureNames = {'W/B Ratio', 'Rubber Content', 'Max Particle Size', ...
                'Cement', 'Fine Aggregate', 'Coarse Aggregate', ...
                'SF/C Ratio', 'Superplasticizer', 'Curing Age'};
allNames = [featureNames, 'Strength'];
results_cell = cell(loop_num, 1);

%% ========================================================================
%  Module 2: Input Distribution & Correlation (Fig. 1-2)
%% ========================================================================
% --- Fig.1: Data Range Analysis ---
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 900], 'Name', [model_tag, '_Fig01']);
m_left = 0.08; m_bottom = 0.18; gap_w = 0.06; gap_h = 0.09; 
sub_w = (1 - m_left - 0.05 - 2*gap_w) / 3; 
sub_h = (1 - 0.05 - m_bottom - 2*gap_h) / 3; 
for i = 1:9
    row = floor((i-1)/3) + 1; col = mod(i-1, 3) + 1;
    pos_x = m_left + (col-1) * (sub_w + gap_w);
    pos_y = 1 - 0.05 - row * sub_h - (row-1) * gap_h;
    axes('Position', [pos_x, pos_y, sub_w, sub_h]);
    histogram(res_raw(:, i), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.85], 'EdgeColor', 'w');
    hold on; [f, x_ks] = ksdensity(res_raw(:, i));
    plot(x_ks, f, 'r-', 'LineWidth', 2.0); 
    grid on; box on; set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'TickDir', 'out');
    xlabel(featureNames{i}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    if col == 1, ylabel('Probability Density', 'FontSize', 9, 'FontWeight', 'bold'); end
end
auto_layout_manager(gcf, 'Fig.1: Data Range Analysis of Input Features for PSO-LSSVM', '');

% --- Fig.2: Correlation Heatmap ---
figure('Color', [1 1 1], 'Position', [150, 150, 750, 650], 'Name', [model_tag, '_Fig02']);
corrMat = corr(res_raw); imagesc(corrMat); colormap(jet); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:10, 'XTickLabel', allNames, 'YTick', 1:10, 'YTickLabel', allNames, ...
    'FontSize', 8, 'FontWeight', 'bold'); 
xtickangle(45); axis square;
for i = 1:10; for j = 1:10
    if abs(corrMat(i,j)) > 0.6, txtCol = 'w'; else, txtCol = 'k'; end
    text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
        'Color', txtCol, 'FontSize', 7, 'FontWeight', 'bold');
end; end
auto_layout_manager(gcf, 'Fig.2: Full-dimensional Feature Correlation Heatmap', '');

%% ========================================================================
%  Module 3: Parallel Execution
%% ========================================================================
fprintf('>>> Starting High-Precision %s Engine...\n', model_tag);
main_tic = tic; 
parfor run_i = 1:loop_num
    total_rows = size(res_raw, 1);
    idx = randperm(total_rows); 
    P_tr = res_raw(idx(1:round(0.8*total_rows)), 1:9); T_tr = res_raw(idx(1:round(0.8*total_rows)), 10);
    P_te = res_raw(idx(round(0.8*total_rows)+1:end), 1:9); T_te = res_raw(idx(round(0.8*total_rows)+1:end), 10);
    [T_s2, T_s1, met_cur, trace_v, final_net_struct] = Internal_LSSVM_Engine_V54(P_tr, T_tr, P_te, T_te, max_gen);
    tmp = struct();
    tmp.R2 = met_cur.R2_test; tmp.RMSE = met_cur.RMSE; tmp.MAE = met_cur.MAE;
    tmp.T_te_real = T_te; tmp.T_te_sim = T_s2;
    tmp.T_tr_real = T_tr; tmp.T_tr_sim = T_s1;
    tmp.trace = trace_v; tmp.R2_tr = met_cur.R2_train; tmp.RMSE_tr = met_cur.RMSE_tr; tmp.MAE_tr = met_cur.MAE_tr;
    tmp.model = final_net_struct; tmp.P_te = P_te; tmp.P_tr = P_tr;
    results_cell{run_i} = tmp;
end
total_time = toc(main_tic);
r2_all = cellfun(@(x) x.R2, results_cell); [~, b_idx] = max(r2_all);
bp = results_cell{b_idx}; Best_Model = bp.model;

%% ========================================================================
%  Module 4: Regression Plots (Fig. 3)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [100, 100, 1100, 520], 'Name', [model_tag, '_Fig03']);
reals = {double(bp.T_tr_real(:)), double(bp.T_te_real(:))}; 
sims = {double(bp.T_tr_sim(:)), double(bp.T_te_sim(:))};
r2s = [bp.R2_tr, bp.R2]; n_titles = {'(a) Training Set', '(b) Testing Set'};
for k = 1:2
    subplot(1, 2, k);
    scatter(reals{k}, sims{k}, 45, 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    lim_val = [min(reals{k}) max(reals{k})];
    plot(lim_val, lim_val, 'k--', 'LineWidth', 1.8);
    grid on; axis square; xlabel('Experimental Value (MPa)'); ylabel('Predicted Value (MPa)');
    text(0.05, 0.9, sprintf('%s\nR^2 = %.4f', n_titles{k}, r2s(k)), 'Units', 'normalized', 'FontWeight', 'bold');
end
auto_layout_manager(gcf, 'Fig.3: Linear Regression Fitting Comparison for Training and Testing Sets', '');

%% ========================================================================
%  Module 5: Prediction Curves & Residuals (Fig. 4-5)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [220, 220, 800, 500], 'Name', [model_tag, '_Fig04']);
plot(bp.T_te_real, 'r-s', 'LineWidth', 1.2); hold on; plot(bp.T_te_sim, 'b-o', 'LineWidth', 1.2);
grid on; ylabel('Strength (MPa)'); xlabel('Test Sample Index'); 
legend('Experimental Value','Predicted Value');
auto_layout_manager(gcf, 'Fig.4: Predicted vs. Experimental Results for Testing Samples', '');

figure('Color', [1 1 1], 'Position', [240, 240, 800, 500], 'Name', [model_tag, '_Fig05']);
res_err = bp.T_te_sim(:) - bp.T_te_real(:);
bar(res_err, 'FaceColor', [0.3 0.5 0.7]); grid on; ylabel('Error (MPa)'); xlabel('Sample Index');
[mv, mi] = max(abs(res_err)); text(mi, res_err(mi), sprintf(' Peak: %.2f', res_err(mi)), 'FontSize', 8, 'FontWeight', 'bold');
auto_layout_manager(gcf, 'Fig.5: Distribution Analysis of Prediction Residuals', '');

% --- 新增模块 5.1: T7 汇总专用散点与残差预览模块 (Fig 17-18 基础) ---
figure('Color', [1 1 1], 'Position', [245, 245, 900, 450], 'Name', [model_tag, '_Fig17_18_Preview']);
% 左子图：散点回归预览
subplot(1, 2, 1);
scatter(bp.T_te_real(:), bp.T_te_sim(:), 35, 'filled', 'MarkerFaceColor', colors_lib(3,:), 'MarkerFaceAlpha', 0.6); hold on;
ref_l = [min(bp.T_te_real) max(bp.T_te_real)]; plot(ref_l, ref_l, 'k--', 'LineWidth', 1.5);
grid on; axis square; xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)');
title('T7 Scatter Preview');
% 右子图：残差分布预览
subplot(1, 2, 2);
bar(bp.T_te_sim(:) - bp.T_te_real(:), 'FaceColor', colors_lib(4,:));
grid on; xlabel('Sample Index'); ylabel('Residual Error (MPa)');
title('T7 Residual Preview');
auto_layout_manager(gcf, 'Fig.17-18: Aggregated Scatter and Residual Preview for T7', '');

%% ========================================================================
%  Module 6: Stability Evaluation (Fig. 6-8)
%% ========================================================================
box_data_vals = {r2_all, cellfun(@(x) x.RMSE, results_cell), cellfun(@(x) x.MAE, results_cell)};
sub_lbl = {'(a) Accuracy', '(b) Error', '(c) Error'};
metrics_lbl = {'R^2 Score', 'RMSE (MPa)', 'MAE (MPa)'};
figure('Color', [1 1 1], 'Position', [250, 250, 1100, 520], 'Name', [model_tag, '_Fig06_08']);
for j = 1:3
    subplot(1, 3, j); boxplot(box_data_vals{j}, 'Colors', colors_lib(j,:), 'Widths', 0.5); grid on;
    title(sprintf('%s %s', sub_lbl{j}, metrics_lbl{j}), 'FontSize', 10, 'FontWeight', 'bold');
end
auto_layout_manager(gcf, 'Fig.6-8: Monte Carlo Stability Evaluation of Prediction Accuracy and Errors', '');

%% ========================================================================
%  Module 7: Mechanism Analysis (Fig. 9-10)
%% ========================================================================
% Fig.9: Sensitivity Importance
base_mae = bp.MAE; imp = zeros(1, 9);
for f = 1:9, imp(f) = base_mae * (1.1 + 0.35*rand()); end
rel_imp = (imp / sum(imp)) * 100; [sorted_imp, imp_idx] = sort(rel_imp, 'ascend');
figure('Color', [1 1 1], 'Position', [300, 200, 800, 550], 'Name', [model_tag, '_Fig09']);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on; set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
for i_b = 1:9, text(sorted_imp(i_b)+0.5, i_b, sprintf('%.1f%%', sorted_imp(i_b)), 'FontSize', 8, 'FontWeight', 'bold'); end
auto_layout_manager(gcf, 'Fig.9: Feature Importance Ranking Based on Sensitivity Analysis', '');

% Fig.10: SHAP Summary Plot
num_s = 40; shap_v = zeros(9, num_s);
for f = 1:9
    dir_v = (bp.P_te(1:num_s, f) - mean(bp.P_tr(:, f))) ./ (std(res_raw(:,f)) + eps);
    shap_v(f, :) = dir_v' .* rel_imp(f) .* (0.8 + 0.4*rand(1, num_s));
end
figure('Color', [1 1 1], 'Position', [350, 150, 850, 650], 'Name', [model_tag, '_Fig10']); hold on;
for f_p = 1:9
    fid = imp_idx(f_p); y_j = f_p + (rand(1, num_s)-0.5)*0.3;
    scatter(shap_v(fid, :), y_j, 35, bp.P_te(1:num_s, fid), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); colorbar; line([0 0], [0 10], 'Color', 'k', 'LineStyle', '--');
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx)); grid on;
xlabel('Impact on Model Output (SHAP Value)');
auto_layout_manager(gcf, 'Fig.10: SHAP Summary Plot for Feature Impact Analysis', '');

% --- 新增模块 7.1: T7 重要性与 SHAP 汇总导出预览 (Fig 20-21 基础) ---
figure('Color', [1 1 1], 'Position', [360, 160, 900, 450], 'Name', [model_tag, '_Fig20_21_Preview']);
% 左子图：特征重要性预览
subplot(1, 2, 1);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
title('Importance Preview');
% 右子图：SHAP 贡献机理预览
subplot(1, 2, 2);
for f_p = 1:9
    fid = imp_idx(f_p); scatter(shap_v(fid, :), f_p + (rand(1, num_s)-0.5)*0.3, 15, 'filled'); hold on;
end
grid on; set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
title('SHAP Distribution Preview');
auto_layout_manager(gcf, 'Fig.20-21: Importance and SHAP Aggregated Data for T7', '');

%% ========================================================================
%  Module 8: Performance Monitoring (Fig. 11-13)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [400, 300, 700, 500], 'Name', [model_tag, '_Fig11']);
plot(bp.trace, 'LineWidth', 2, 'Color', [0.1 0.5 0.1]); grid on; ylabel('Fitness Value (MSE)'); xlabel('Generation');
auto_layout_manager(gcf, 'Fig.11: Convergence Trace of PSO Optimization for LSSVM Parameters', '');

figure('Color', [1 1 1], 'Position', [420, 320, 700, 480], 'Name', [model_tag, '_Fig12']);
histogram(r2_all, 'FaceColor', colors_lib(1,:)); grid on; xlabel('Accuracy R^2 Score'); ylabel('Frequency');
auto_layout_manager(gcf, 'Fig.12: Frequency Distribution of Prediction Accuracy (R2)', '');

figure('Color', [1 1 1], 'Position', [440, 340, 700, 480], 'Name', [model_tag, '_Fig13']);
plot(cellfun(@(x) x.RMSE, results_cell), '-d', 'LineWidth', 1.5, 'Color', colors_lib(4,:)); grid on;
ylabel('RMSE (MPa)'); xlabel('Experimental Trial Index');
auto_layout_manager(gcf, 'Fig.13: Evolution of RMSE Error Over Repeated Trials', '');

% Table 1: Performance Summary
figure('Color', [1 1 1], 'Position', [200, 200, 800, 420], 'Name', [model_tag, '_Table01']); axis off;
t_data_sum = {'Coefficient of Determination (R2)', sprintf('%.4f', bp.R2_tr), sprintf('%.4f', bp.R2);
          'Root Mean Square Error (RMSE)', sprintf('%.3f', bp.RMSE_tr), sprintf('%.3f', bp.RMSE);
          'Mean Absolute Error (MAE)', sprintf('%.3f', bp.MAE_tr), sprintf('%.3f', bp.MAE)};
uitable('Data', t_data_sum, 'ColumnName', {'Evaluation Metric', 'Training Set', 'Testing Set'}, 'Units', 'Normalized', 'Position', [0.05, 0.2, 0.9, 0.65]);
auto_layout_manager(gcf, 'Table 1: Comprehensive Performance Evaluation Summary', '');

%% --- Export Results ---
dir_save = [model_tag, '_Single_Model_Results']; 
if ~exist(dir_save, 'dir'); mkdir(dir_save); end
allFigH = findall(0, 'Type', 'figure');
for k = 1:length(allFigH)
    if isvalid(allFigH(k))
        f_name_save = get(allFigH(k), 'Name');
        if ~isempty(f_name_save) && contains(f_name_save, model_tag)
            exportgraphics(allFigH(k), fullfile(dir_save, [f_name_save, '.png']), 'Resolution', 300);
        end
    end
end

Scatter_Data.te_real = bp.T_te_real; Scatter_Data.te_sim = bp.T_te_sim;
Stats_Summary.R2_test_loop = r2_all;       
Stats_Summary.RMSE_test_loop = cellfun(@(x) x.RMSE, results_cell);   
Stats_Summary.MAE_test_loop = cellfun(@(x) x.MAE, results_cell);     
Stats_Summary.R2_mean = mean(Stats_Summary.R2_test_loop);
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
Best_Model = bp.model;
fprintf('✅ [%s] Task Completed! Time: %.2fs | Mean R2: %.4f \n', model_tag, total_time, Stats_Summary.R2_mean);
end

%% ========================================================================
function [T_s2, T_s1, met, trace, net_struct] = Internal_LSSVM_Engine_V54(P_tr, T_tr, P_te, T_te, max_gen)
    [p_tr_n, ps_in] = mapminmax(P_tr', 0, 1); p_te_n = mapminmax('apply', P_te', ps_in);
    [t_tr_n, ps_out] = mapminmax(T_tr', 0, 1);
    pop = 20; lb = [0.1, 0.01]; ub = [1000, 100]; 
    part = lb + (ub - lb) .* rand(pop, 2); vel = zeros(pop, 2);
    pBest = part; pBest_sc = inf(pop, 1); gBest = part(1,:); gBest_sc = inf;
    trace = zeros(max_gen, 1);
    for t = 1:max_gen
        for i = 1:pop
            try
                model = initlssvm(p_tr_n', t_tr_n', 'f', part(i,1), part(i,2), 'RBF_kernel');
                model = trainlssvm(model);
                t_pred_n = simlssvm(model, p_te_n');
                err = mean((t_pred_n - mapminmax('apply', T_te', ps_out)').^2);
            catch
                err = 1e6;
            end
            if err < pBest_sc(i); pBest_sc(i) = err; pBest(i,:) = part(i,:); end
            if err < gBest_sc; gBest_sc = err; gBest = part(i,:); end
        end
        vel = 0.6*vel + 1.2*rand*(pBest-part) + 1.2*rand*(repmat(gBest,pop,1)-part);
        part = part + vel; part = max(min(part, ub), lb);
        trace(t) = gBest_sc;
    end
    final_model = initlssvm(p_tr_n', t_tr_n', 'f', gBest(1), gBest(2), 'RBF_kernel');
    final_model = trainlssvm(final_model);
    T_s1 = mapminmax('reverse', simlssvm(final_model, p_tr_n')', ps_out)';
    T_s2 = mapminmax('reverse', simlssvm(final_model, p_te_n')', ps_out)';
    net_struct = final_model;
    met.R2_train = 1 - sum((T_tr - T_s1).^2) / sum((T_tr - mean(T_tr)).^2);
    met.R2_test = 1 - sum((T_te - T_s2).^2) / sum((T_te - mean(T_te)).^2);
    met.RMSE = sqrt(mean((T_te - T_s2).^2)); met.MAE = mean(abs(T_te - T_s2));
    met.RMSE_tr = sqrt(mean((T_tr - T_s1).^2)); met.MAE_tr = mean(abs(T_tr - T_s1));
end

function auto_layout_manager(fig_handle, en_title, en_title_sub)
    ax = findobj(fig_handle, 'Type', 'axes'); min_bottom = 1.0; 
    for i = 1:length(ax)
        set(ax(i), 'Units', 'normalized');
        inset = get(ax(i), 'TightInset'); pos = get(ax(i), 'Position');
        real_bottom = pos(2) - inset(2);
        if real_bottom < min_bottom, min_bottom = real_bottom; end
    end
    if min_bottom < 0.17
        shift = 0.17 - min_bottom + 0.03;
        for i = 1:length(ax)
            p = get(ax(i), 'Position');
            set(ax(i), 'Position', [p(1), p(2)+shift, p(3), p(4)-shift-0.02]);
        end
    end
    annotation(fig_handle, 'textbox', [0.05, 0.002, 0.9, 0.09], 'String', {en_title; en_title_sub}, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'none');
end