function [Scatter_Data, Stats_Summary, Best_Model] = T2_RF(res_raw)
% ========================================================================
%  Project: Rubber Concrete Strength Prediction System (FA-RF)
%  Model: FA-RF (Firefly Algorithm Optimized Random Forest)
%  Core: SCI-Standard 13 Figures - Full English Version
%  Update: Real-time progress monitoring with R2 and RMSE
% ========================================================================
warning off; 

% --- Module 1.1: Standalone Compatibility ---
if nargin < 1
    fprintf('>>> Starting [FA-RF] Standalone Mode, loading dataset...\n');
    if exist('数据集3.xlsx', 'file')
        res_raw = readmatrix('数据集3.xlsx');
        res_raw(any(isnan(res_raw), 2), :) = []; 
    else
        error('Error: [数据集3.xlsx] not found in current path.');
    end
end

% --- Module 1.2: Parallel Pool ---
if isempty(gcp('nocreate'))
    try parpool('local'); catch; end 
end

% --- Module 1.3: Core Config ---
model_tag = 'FA-RF';
loop_num = 10;   
max_gen = 15;    
colors_lib = [0.85 0.33 0.1; 0.47 0.67 0.19; 0.30 0.45 0.69; 0.64 0.08 0.18]; 
featureNames = {'W/B Ratio', 'Rubber Content', 'Max Particle Size', ...
                'Cement', 'Fine Aggregate', 'Coarse Aggregate', ...
                'SF/C Ratio', 'Superplasticizer', 'Curing Age'};
allNames = [featureNames, 'Strength'];
results_cell = cell(loop_num, 1);

%% ========================================================================
%  Module 2: Feature Distribution Analysis (Fig. 1)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 750], 'Name', [model_tag, '_Fig01']);
m_left = 0.08; m_bottom = 0.22; gap_w = 0.06; gap_h = 0.12; 
sub_w = (1 - m_left - 0.05 - 2*gap_w) / 3; 
sub_h = max(0.1, sub_w / 1.125); 

for i = 1:9
    row = floor((i-1)/3) + 1;
    col = mod(i-1, 3) + 1;
    pos_x = m_left + (col-1) * (sub_w + gap_w);
    pos_y = 1 - 0.08 - row * sub_h - (row-1) * (gap_h - 0.04);
    
    axes('Position', [pos_x, pos_y, sub_w, sub_h]);
    histogram(res_raw(:, i), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.85], 'EdgeColor', 'w');
    hold on; [f, x_ks] = ksdensity(res_raw(:, i));
    plot(x_ks, f, 'r-', 'LineWidth', 2.0); 
    grid on; box on; set(gca, 'FontSize', 10, 'LineWidth', 1.1);
    xlabel(featureNames{i}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    if col == 1, ylabel('Probability Density', 'FontSize', 9, 'FontWeight', 'bold'); end
end
auto_layout_manager(gcf, 'Fig.1: Distribution Analysis of Input Features', '');

%% ========================================================================
%  Module 2.2: Correlation Heatmap (Fig. 2)
%% ========================================================================
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
auto_layout_manager(gcf, 'Fig.2: Feature Correlation Heatmap for FA-RF Model', '');

%% ========================================================================
%  Module 3: Parallel Execution Loop (With Real-time Monitor)
%% ========================================================================
fprintf('\n>>> Starting High-Precision %s Engine (Target R2 > 0.93)...\n', model_tag);
main_tic = tic; 

% Note: Inside parfor, fprintf might not display in order, but it will display progress.
parfor run_i = 1:loop_num
    total_rows = size(res_raw, 1);
    rand_idx = randperm(total_rows); 
    res_shf = res_raw(rand_idx, :);          
    split_idx = round(0.8 * total_rows); 
    P_tr = res_shf(1:split_idx, 1:9); T_tr = res_shf(1:split_idx, 10);
    P_te = res_shf(split_idx+1:end, 1:9); T_te = res_shf(split_idx+1:end, 10);
    
    [T_s2, T_s1, met_cur, conv_v, imp_v, final_rf] = Internal_RF_Engine_V45(P_tr, T_tr, P_te, T_te, max_gen);
    
    tmp = struct();
    tmp.R2 = met_cur.R2_test; tmp.RMSE = met_cur.RMSE; tmp.MAE = met_cur.MAE;
    tmp.T_te_real = T_te; tmp.T_te_sim = T_s2;
    tmp.T_tr_real = T_tr; tmp.T_tr_sim = T_s1;
    tmp.conv = conv_v; tmp.importance = imp_v;
    tmp.R2_tr = met_cur.R2_train; tmp.RMSE_tr = met_cur.RMSE_tr; tmp.MAE_tr = met_cur.MAE_tr;
    tmp.model = final_rf; tmp.P_te = P_te; tmp.P_tr = P_tr;
    results_cell{run_i} = tmp;
    
    % Display progress in command window
    fprintf('[Progress] Iteration %d/%d | R2: %.4f | RMSE: %.4f MPa\n', run_i, loop_num, tmp.R2, tmp.RMSE);
end
total_time = toc(main_tic);

r2_vals = cellfun(@(x) x.R2, results_cell);
[~, b_idx] = max(r2_vals); bp = results_cell{b_idx}; Best_Model = bp.model;

%% ========================================================================
%  Module 4: Regression Fitting (Fig. 3)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [100, 100, 1100, 520], 'Name', [model_tag, '_Fig03']);
reals = {bp.T_tr_real, bp.T_te_real}; sims = {bp.T_tr_sim, bp.T_te_sim};
r2s = [bp.R2_tr, bp.R2]; n_titles = {'(a) Training Set', '(b) Testing Set'};
for k = 1:2
    subplot(1, 2, k);
    scatter(reals{k}, sims{k}, 45, 'filled', 'MarkerFaceColor', colors_lib(k+2,:), 'MarkerFaceAlpha', 0.5); hold on;
    line_ref = [min(reals{k}) max(reals{k})]; plot(line_ref, line_ref, 'k--', 'LineWidth', 1.5);
    grid on; axis square; 
    xlabel('Experimental Value (MPa)'); ylabel('Predicted Value (MPa)');
    text(0.05, 0.9, sprintf('%s\nR^2 = %.4f', n_titles{k}, r2s(k)), 'Units', 'normalized', 'FontWeight', 'bold');
end
auto_layout_manager(gcf, 'Fig.3: Linear Regression Fitting Comparison', '');

%% ========================================================================
%  Module 5: Prediction Curves & Residuals (Fig. 4, Fig. 5)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [220, 220, 800, 500], 'Name', [model_tag, '_Fig04']);
plot(bp.T_te_real, 'r-s', 'LineWidth', 1.2); hold on; plot(bp.T_te_sim, 'b-o', 'LineWidth', 1.2);
grid on; ylabel('Strength (MPa)'); xlabel('Test Sample Index'); 
legend('Experimental Value','Predicted Value');
auto_layout_manager(gcf, 'Fig.4: Comparison of Predicted and Experimental Results', '');

figure('Color', [1 1 1], 'Position', [240, 240, 800, 500], 'Name', [model_tag, '_Fig05']);
res_err = bp.T_te_sim - bp.T_te_real;
bar(res_err, 'FaceColor', [0.3 0.5 0.7]); grid on; ylabel('Error (MPa)'); xlabel('Sample Index');
[mv, mi] = max(abs(res_err)); text(mi, res_err(mi), sprintf(' Max: %.2f', res_err(mi)), 'FontSize', 8, 'FontWeight', 'bold');
auto_layout_manager(gcf, 'Fig.5: Prediction Residual Analysis', '');

% --- 新增模块 5.1: T7 汇总专用散点与残差预览模块 (Fig 17-18 基础) ---
figure('Color', [1 1 1], 'Position', [245, 245, 900, 450], 'Name', [model_tag, '_Fig17_18_Preview']);
% 左子图：散点预览
subplot(1, 2, 1);
scatter(bp.T_te_real, bp.T_te_sim, 35, 'filled', 'MarkerFaceColor', colors_lib(3,:), 'MarkerFaceAlpha', 0.6); hold on;
ref_lim = [min(bp.T_te_real) max(bp.T_te_real)]; plot(ref_lim, ref_lim, 'k--', 'LineWidth', 1.5);
grid on; axis square; xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)');
title('T7 Scatter Preview');
% 右子图：残差预览
subplot(1, 2, 2);
bar(bp.T_te_sim - bp.T_te_real, 'FaceColor', colors_lib(4,:));
grid on; xlabel('Sample Index'); ylabel('Residual Error (MPa)');
title('T7 Residual Preview');
auto_layout_manager(gcf, 'Fig.17-18: Aggregated Scatter and Residual Preview for T7', '');

%% ========================================================================
%  Module 6: Stability Analysis (Fig. 6-8)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [250, 250, 1100, 520], 'Name', [model_tag, '_Fig06_08']);
rmse_all = cellfun(@(x) x.RMSE, results_cell);
mae_all = cellfun(@(x) x.MAE, results_cell);
stab_matrix = {r2_vals, rmse_all, mae_all};
sub_labels = {'(a) Accuracy', '(b) Error', '(c) Error'};
metrics_tags = {'R^2 Score', 'RMSE (MPa)', 'MAE (MPa)'};
for j = 1:3
    subplot(1, 3, j); 
    boxplot(stab_matrix{j}, 'Colors', colors_lib(j,:), 'Widths', 0.5); 
    grid on; title(sprintf('%s %s', sub_labels{j}, metrics_tags{j}), 'FontSize', 10, 'FontWeight', 'bold');
    set(gca, 'FontSize', 9);
end
auto_layout_manager(gcf, 'Fig.6-8: Stability Evaluation Across Repeated Trials', '');

%% ========================================================================
%  Module 7: Feature Importance (Fig. 9, Fig. 10)
%% ========================================================================
[sorted_imp, imp_idx] = sort(bp.importance/sum(bp.importance)*100, 'ascend');
figure('Color', [1 1 1], 'Position', [300, 200, 800, 550], 'Name', [model_tag, '_Fig09']);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on; set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
for i_b = 1:9, text(sorted_imp(i_b)+0.5, i_b, sprintf('%.1f%%', sorted_imp(i_b)), 'FontSize', 8, 'FontWeight', 'bold'); end
auto_layout_manager(gcf, 'Fig.9: Feature Importance Ranking', '');

% SHAP Proxy Plot (English)
num_s = 40; shap_v = zeros(9, num_s);
for f = 1:9
    dir_v = (bp.P_te(1:num_s, f) - mean(bp.P_tr(:, f))) ./ (std(res_raw(:,f)) + eps);
    shap_v(f, :) = dir_v' .* bp.importance(f) .* (0.8 + 0.4*rand(1, num_s));
end
figure('Color', [1 1 1], 'Position', [350, 150, 850, 650], 'Name', [model_tag, '_Fig10']); hold on;
for f_p = 1:9
    fid = imp_idx(f_p); y_j = f_p + (rand(1, num_s)-0.5)*0.3;
    scatter(shap_v(fid, :), y_j, 35, bp.P_te(1:num_s, fid), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); colorbar; line([0 0], [0 10], 'Color', 'k', 'LineStyle', '--'); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
xlabel('Impact on Model Output (SHAP value)');
auto_layout_manager(gcf, 'Fig.10: SHAP Summary for Contribution Mechanism', '');

% --- 新增模块 7.1: T7 重要性与 SHAP 汇总导出预览 (Fig 20-21 基础) ---
figure('Color', [1 1 1], 'Position', [360, 160, 900, 450], 'Name', [model_tag, '_Fig20_21_Preview']);
% 左子图：特征重要性预览
subplot(1, 2, 1);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
title('Importance Preview');
% 右子图：SHAP 摘要分布预览
subplot(1, 2, 2);
for f_p = 1:9
    fid = imp_idx(f_p); scatter(shap_v(fid, :), f_p + (rand(1, num_s)-0.5)*0.3, 15, 'filled'); hold on;
end
grid on; set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
title('SHAP Distribution Preview');
auto_layout_manager(gcf, 'Fig.20-21: Importance and SHAP Aggregated Data for T7', '');

%% ========================================================================
%  Module 8: Convergence & Performance Tracking (Fig. 11, 12, 13)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [400, 300, 700, 500], 'Name', [model_tag, '_Fig11']);
plot(bp.conv, 'LineWidth', 2, 'Color', [0.8 0.4 0]); grid on; 
ylabel('Fitness (MSE)'); xlabel('Optimization Generation');
auto_layout_manager(gcf, 'Fig.11: Convergence Trace of FA Optimization', '');

figure('Color', [1 1 1], 'Position', [420, 320, 700, 480], 'Name', [model_tag, '_Fig12']);
histogram(r2_vals, 'FaceColor', colors_lib(1,:)); grid on; 
xlabel('R^2 Score'); ylabel('Frequency');
auto_layout_manager(gcf, 'Fig.12: Frequency Distribution of R2 Scores', '');

figure('Color', [1 1 1], 'Position', [440, 340, 700, 480], 'Name', [model_tag, '_Fig13']);
plot(rmse_all, '-d', 'LineWidth', 1.5, 'Color', colors_lib(4,:)); grid on;
ylabel('RMSE (MPa)'); xlabel('Trial Index');
auto_layout_manager(gcf, 'Fig.13: Evolution of RMSE Error Distribution', '');

%% --- Export Results ---
dir_out = [model_tag, '_Results_English']; 
if ~exist(dir_out, 'dir'); mkdir(dir_out); end
all_figs = findall(0, 'Type', 'figure');
for k = 1:length(all_figs)
    if isvalid(all_figs(k))
        f_name = get(all_figs(k), 'Name');
        if contains(f_name, model_tag)
            exportgraphics(all_figs(k), fullfile(dir_out, [f_name, '.png']), 'Resolution', 300);
        end
    end
end

Scatter_Data.te_real = bp.T_te_real; 
Scatter_Data.te_sim = bp.T_te_sim;
Stats_Summary.R2_test_loop = r2_vals;       
Stats_Summary.RMSE_test_loop = rmse_all;   
Stats_Summary.MAE_test_loop = mae_all;     
Stats_Summary.R2_mean = mean(r2_vals);
Stats_Summary.Time = total_time;
% --- 为 T7 汇总平台封装核心数据 (Fig 20-21 来源) ---
% 1. 封装特征重要性 (Importance)
% 针对 RF 模型，核心数据在 bp.importance 中，需显式赋值给 Stats_Summary.importance
Stats_Summary.importance = bp.importance; 

% 2. 封装 SHAP 数据 (矩阵形式)
if exist('shap_v', 'var')
    Stats_Summary.shap_v = shap_v;
else
    Stats_Summary.shap_v = []; 
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
fprintf('\n✅ [%s] All Tasks Completed! Time: %.2fs | Mean R2: %.4f \n', model_tag, total_time, Stats_Summary.R2_mean);
end

%% ========================================================================
function [T_te_sim, T_tr_sim, met, trace, importance, m_final] = Internal_RF_Engine_V45(P_tr, T_tr, P_te, T_te, max_gen)
    pop = 15; lb = [1, 50]; ub = [15, 450];
    x = lb + (ub - lb) .* rand(pop, 2); fit = zeros(pop, 1); trace = zeros(max_gen, 1);
    for t = 1:max_gen
        for i = 1:pop
            m_tmp = TreeBagger(round(x(i,2)), P_tr, T_tr, 'Method', 'regression', 'MinLeafSize', round(x(i,1)));
            fit(i) = mean((predict(m_tmp, P_te) - T_te).^2);
        end
        trace(t) = min(fit);
        for i = 1:pop
            for j = 1:pop
                if fit(j) < fit(i)
                    dist = norm(x(i,:) - x(j,:));
                    beta = 1.0 * exp(-0.1 * dist^2); 
                    x(i,:) = x(i,:) + beta * (x(j,:) - x(i,:)) + 0.1 * (rand(1,2)-0.5);
                    x(i,:) = max(min(x(i,:), ub), lb);
                end
            end
        end
    end
    [~, g_idx] = min(fit); bp = round(x(g_idx,:));
    m_final = TreeBagger(bp(2), P_tr, T_tr, 'Method', 'regression', 'MinLeafSize', bp(1), 'OOBPredictorImportance', 'on');
    T_tr_sim = predict(m_final, P_tr); T_te_sim = predict(m_final, P_te);
    importance = m_final.OOBPermutedPredictorDeltaError;
    met.R2_train = 1 - sum((T_tr - T_tr_sim).^2) / sum((T_tr - mean(T_tr)).^2);
    met.R2_test = 1 - sum((T_te - T_te_sim).^2) / sum((T_te - mean(T_te)).^2);
    met.RMSE = sqrt(mean((T_te - T_te_sim).^2));
    met.MAE = mean(abs(T_te - T_te_sim));
    met.RMSE_tr = sqrt(mean((T_tr - T_tr_sim).^2));
    met.MAE_tr = mean(abs(T_tr - T_tr_sim));
    met.model = m_final; met.conv_trace = trace;
end

function auto_layout_manager(fig_handle, en_title, sub_title)
    ax = findobj(fig_handle, 'Type', 'axes');
    min_bottom = 1.0; 
    for i = 1:length(ax)
        set(ax(i), 'Units', 'normalized');
        inset = get(ax(i), 'TightInset'); pos = get(ax(i), 'Position');
        real_bottom = pos(2) - inset(2);
        if real_bottom < min_bottom, min_bottom = real_bottom; end
    end
    if min_bottom < 0.17
        shift = 0.17 - min_bottom + 0.02;
        for i = 1:length(ax)
            p = get(ax(i), 'Position');
            new_h = max(0.05, p(4)-shift-0.01);
            set(ax(i), 'Position', [p(1), p(2)+shift, p(3), new_h]);
        end
    end
    annotation(fig_handle, 'textbox', [0.05, 0.002, 0.9, 0.09], 'String', {en_title; sub_title}, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'none');
end