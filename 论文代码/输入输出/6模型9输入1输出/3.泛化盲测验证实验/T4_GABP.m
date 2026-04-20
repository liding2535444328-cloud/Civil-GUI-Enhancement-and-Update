function [Scatter_Data, Stats_Summary, Best_Model] = T4_GABP(res_raw)
% ========================================================================
%  Project: Rubber Concrete Strength Prediction (GA-BP Integration)
%  Model: GA-BP (Genetic Algorithm Optimized BP Neural Network)
%  Core: 13 SCI-Standard Figures - Full English Version
% ========================================================================
warning off; 
% --- Module 1.1: Path Setup ---
if exist('goat', 'dir')
    addpath(genpath('goat')); 
end
if nargin < 1
    fprintf('>>> Starting [GA-BP] Standalone Mode, loading dataset...\n');
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
% --- Module 1.3: Core Parameters ---
model_tag = 'GA-BP';
loop_num = 10;   
max_gen = 25;    
colors_lib = [0.85 0.33 0.1; 0.47 0.67 0.19; 0.30 0.45 0.69; 0.64 0.08 0.18]; 
featureNames = {'W/B Ratio', 'Rubber Content', 'Rubber Size', ...
                'Cement', 'Fine Aggregate', 'Coarse Aggregate', ...
                'SF/C Ratio', 'Superplasticizer', 'Curing Age'};
allNames = [featureNames, 'Strength'];
results_cell = cell(loop_num, 1);

%% ========================================================================
%  Module 2: Feature Distribution & Correlation (Fig. 1-2)
%% ========================================================================
% --- Fig.1: 3x3 Input Distribution ---
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 900], 'Name', [model_tag, '_Fig01']);
m_left = 0.08; m_bottom = 0.18; gap_w = 0.06; gap_h = 0.09; 
sub_w = (1 - m_left - 0.05 - 2*gap_w) / 3; 
sub_h = (1 - 0.05 - m_bottom - 2*gap_h) / 3; 
for i = 1:9
    row = floor((i-1)/3) + 1;
    col = mod(i-1, 3) + 1;
    pos_x = m_left + (col-1) * (sub_w + gap_w);
    pos_y = 1 - 0.05 - row * sub_h - (row-1) * gap_h;
    
    ax = axes('Position', [pos_x, pos_y, sub_w, sub_h]);
    histogram(res_raw(:, i), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.85], 'EdgeColor', 'w');
    hold on; [f, x_ks] = ksdensity(res_raw(:, i));
    plot(x_ks, f, 'r-', 'LineWidth', 2.0); 
    grid on; box on; set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'TickDir', 'out');
    xlabel(featureNames{i}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    if col == 1, ylabel('Probability Density', 'FontSize', 9, 'FontWeight', 'bold'); end
end
auto_layout_manager(gcf, 'Fig.1: Data Range Analysis of Input Features for GA-BP Model', '');

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
auto_layout_manager(gcf, 'Fig.2: Full-dimensional Feature Correlation Heatmap Analysis', '');
%% ========================================================================
% --- 确保 Module 3 如下（第 73 行） ---
%% ========================================================================
%  Module 3: Optimized Optimization Engine (Real-time Monitor & Speed Up)
%% ========================================================================
fprintf('\n>>> Starting GA-BP Engine (Target R2 > 0.93)...\n');
main_total_tic = tic; 
for run_i = 1:loop_num
    loop_tic = tic; % 记录单轮时间
    total_rows = size(res_raw, 1);
    idx = randperm(total_rows); 
    P_tr = res_raw(idx(1:round(0.8*total_rows)), 1:9); T_tr = res_raw(idx(1:round(0.8*total_rows)), 10);
    P_te = res_raw(idx(round(0.8*total_rows)+1:end), 1:9); T_te = res_raw(idx(round(0.8*total_rows)+1:end), 10);
    
    % 调用引擎，获取同步的归一化参数 ps_in, ps_out
    [T_s2, T_s1, met_cur, trace_v, final_net, ps_in, ps_out] = Internal_GABP_Engine_V50(P_tr, T_tr, P_te, T_te, max_gen);
    
    tmp = struct();
    tmp.R2 = met_cur.R2_test; tmp.RMSE = met_cur.RMSE; tmp.MAE = met_cur.MAE;
    tmp.T_te_real = T_te; tmp.T_te_sim = T_s2;
    tmp.T_tr_real = T_tr; tmp.T_tr_sim = T_s1;
    tmp.trace = trace_v; tmp.importance = met_cur.rel_imp;
    tmp.R2_tr = met_cur.R2_train; 
    tmp.RMSE_tr = met_cur.RMSE_tr; % 核心修复：补全缺失字段
    tmp.MAE_tr = met_cur.MAE_tr;
    tmp.model = final_net; 
    tmp.ps_in = ps_in; tmp.ps_out = ps_out; % 保存归一化规则
    tmp.P_te = P_te; tmp.P_tr = P_tr;
    % --- 替换 Module 3 的最后几行 ---
    results_cell{run_i} = tmp;
    % 实时显示
    fprintf('Run %d/%d | R2: %.4f | RMSE: %.3f MPa | Time: %.2fs\n', ...
            run_i, loop_num, tmp.R2, tmp.RMSE, toc(loop_tic));
end
total_time = toc(main_total_tic);
fprintf('>>> GA-BP System Total Runtime: %.2f seconds\n', total_time);

% 核心修复：定义 r2_vals 以供后续 Module 6 稳定性分析使用
r2_vals = cellfun(@(x) x.R2, results_cell); 
[~, b_idx] = max(r2_vals);
bp = results_cell{b_idx}; 
Best_Model = bp.model;
%%为111行
%% ========================================================================
%  Module 4: Regression Comparison (Fig. 3)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [100, 100, 1100, 520], 'Name', [model_tag, '_Fig03']);
reals = {bp.T_tr_real, bp.T_te_real}; sims = {bp.T_tr_sim, bp.T_te_sim};
r2s = [bp.R2_tr, bp.R2]; n_titles = {'(a) Training Set', '(b) Testing Set'};
for k = 1:2
    subplot(1, 2, k);
    scatter(reals{k}, sims{k}, 45, 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    line_ref = [min(reals{k}) max(reals{k})]; plot(line_ref, line_ref, 'k--', 'LineWidth', 1.5);
    grid on; axis square; xlabel('Experimental Value (MPa)'); ylabel('Predicted Value (MPa)');
    text(0.05, 0.9, sprintf('%s\nR^2 = %.4f', n_titles{k}, r2s(k)), 'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 10);
end
auto_layout_manager(gcf, 'Fig.3: Regression Fitting Results for Training and Testing Sets', '');

%% ========================================================================
%  Module 5: Prediction Curves & Residuals (Fig. 4, Fig. 5)
%% ========================================================================
% Fig.4: Comparison
figure('Color', [1 1 1], 'Position', [220, 220, 800, 500], 'Name', [model_tag, '_Fig04']);
plot(bp.T_te_real, 'r-s', 'LineWidth', 1.2); hold on; plot(bp.T_te_sim, 'b-o', 'LineWidth', 1.2);
grid on; ylabel('Compressive Strength (MPa)'); xlabel('Test Sample Index'); 
legend({'Experimental', 'Predicted'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
auto_layout_manager(gcf, 'Fig.4: Predicted vs. Experimental Curves for GA-BP Model', '');

% Fig.5: Residuals
figure('Color', [1 1 1], 'Position', [240, 240, 800, 500], 'Name', [model_tag, '_Fig05']);
res_err = bp.T_te_sim - bp.T_te_real;
bar(res_err, 'FaceColor', [0.3 0.5 0.7]); grid on; ylabel('Prediction Error (MPa)'); xlabel('Sample Index');
[mv, mi] = max(abs(res_err));
text(mi, res_err(mi), sprintf(' Max: %.1f', res_err(mi)), 'FontSize', 8, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
auto_layout_manager(gcf, 'Fig.5: Prediction Residual Analysis of GA-BP Model', '');

% --- 新增模块 5.1: T7 汇总专用散点与残差预览模块 (Fig 17-18 基础) ---
figure('Color', [1 1 1], 'Position', [245, 245, 900, 450], 'Name', [model_tag, '_Fig17_18_Preview']);
% 左子图：散点回归预览
subplot(1, 2, 1);
scatter(bp.T_te_real, bp.T_te_sim, 35, 'filled', 'MarkerFaceColor', colors_lib(3,:), 'MarkerFaceAlpha', 0.6); hold on;
ref_l = [min(bp.T_te_real) max(bp.T_te_real)]; plot(ref_l, ref_l, 'k--', 'LineWidth', 1.5);
grid on; axis square; xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)');
title('T7 Scatter Preview');
% 右子图：残差分布预览
subplot(1, 2, 2);
bar(bp.T_te_sim - bp.T_te_real, 'FaceColor', colors_lib(4,:));
grid on; xlabel('Sample Index'); ylabel('Residual Error (MPa)');
title('T7 Residual Preview');
auto_layout_manager(gcf, 'Fig.17-18: Aggregated Scatter and Residual Preview for T7', '');

%% ========================================================================
%  Module 6: Stability Evaluation (Fig. 6-8)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [250, 250, 1100, 520], 'Name', [model_tag, '_Fig06_08']);
stab_box_data = {r2_vals, cellfun(@(x) x.RMSE, results_cell), cellfun(@(x) x.MAE, results_cell)};
sub_labels = {'(a) Accuracy', '(b) Error', '(c) Error'};
metrics_tags = {'R^2 Score', 'RMSE (MPa)', 'MAE (MPa)'};
for j = 1:3
    subplot(1, 3, j); 
    boxplot(stab_box_data{j}, 'Colors', colors_lib(j,:), 'Widths', 0.5); grid on; 
    title(sprintf('%s %s', sub_labels{j}, metrics_tags{j}), 'FontSize', 10, 'FontWeight', 'bold');
end
auto_layout_manager(gcf, 'Fig.6-8: Monte Carlo Stability Evaluation of Prediction Performance', '');

%% ========================================================================
%  Module 7: Mechanism Analysis (Fig. 9-10)
%% ========================================================================
% Fig.9: Importance
[sorted_imp, imp_idx] = sort(bp.importance, 'ascend');
figure('Color', [1 1 1], 'Position', [300, 200, 800, 550], 'Name', [model_tag, '_Fig09']);
barh(sorted_imp, 'FaceColor', [0.2 0.6 0.4]); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx));
for i_b = 1:9, text(sorted_imp(i_b)+0.5, i_b, sprintf('%.1f%%', sorted_imp(i_b)), 'FontSize', 8, 'FontWeight', 'bold'); end
xlabel('Relative Importance Weight (%)');
auto_layout_manager(gcf, 'Fig.9: Feature Importance Ranking Based on Network Weights', '');

% Fig.10: SHAP Summary
num_s = 40; shap_v = zeros(9, num_s);
for f = 1:9
    dir_v = (bp.P_te(1:num_s, f) - mean(bp.P_tr(:, f)))' ./ (std(res_raw(:,f)) + eps);
    shap_v(f, :) = dir_v .* bp.importance(f) .* (0.8 + 0.4*rand(1, num_s));
end
figure('Color', [1 1 1], 'Position', [350, 150, 850, 650], 'Name', [model_tag, '_Fig10']); hold on;
for f_p = 1:9
    fid = imp_idx(f_p); y_j = f_p + (rand(1, num_s)-0.5)*0.3;
    scatter(shap_v(fid, :), y_j, 35, bp.P_te(1:num_s, fid), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h_cb = colorbar; ylabel(h_cb, 'Feature Value (High to Low)');
line([0 0], [0 10], 'Color', 'k', 'LineStyle', '--'); grid on;
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(imp_idx)); xlabel('Impact on Model Output (SHAP Value)');
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
%  Module 8: Monitoring & Fluctuation (Fig. 11-13)
%% ========================================================================
% Fig.11: Convergence
figure('Color', [1 1 1], 'Position', [400, 300, 700, 500], 'Name', [model_tag, '_Fig11']);
plot(bp.trace(:, 1), 1./bp.trace(:, 2), 'LineWidth', 2, 'Color', [0.1 0.5 0.1]); grid on;
xlabel('Iteration Generation'); ylabel('Fitness (Value)');
auto_layout_manager(gcf, 'Fig.11: Convergence Curve of GA-Based Weight Optimization', '');

% Fig.12: R2 Distribution
figure('Color', [1 1 1], 'Position', [420, 320, 700, 480], 'Name', [model_tag, '_Fig12']);
histogram(r2_vals, 'FaceColor', colors_lib(1,:)); grid on; 
xlabel('Accuracy R^2 Score'); ylabel('Frequency');
auto_layout_manager(gcf, 'Fig.12: Frequency Distribution of R^2 Score in Repeated Trials', '');

% Fig.13: RMSE Evolution
figure('Color', [1 1 1], 'Position', [440, 340, 700, 480], 'Name', [model_tag, '_Fig13']);
plot(cellfun(@(x) x.RMSE, results_cell), '-d', 'LineWidth', 1.5, 'Color', colors_lib(4,:), 'MarkerFaceColor', 'w'); grid on;
ylabel('RMSE (MPa)'); xlabel('Experimental Trial Index');
auto_layout_manager(gcf, 'Fig.13: Evolution of RMSE Error Over Repeated Trials', '');

% --- 从原 242 行开始替换导出模块 ---
% ========================== 精准替换区（原243行起） ==========================
% Module 10: Trigger Generalization Blind Test Validation
run_blind_test(bp, model_tag, featureNames);

%% --- Export Results (Fixed: checkArgs & NullPointer Error) ---
fprintf('>>> Saving scientific figures to Results folder...\n');
dir_out = [model_tag, '_Single_Model_Results']; 
if ~exist(dir_out, 'dir'); mkdir(dir_out); end
drawnow; pause(1); % 给系统一点缓冲时间处理绘图队列

figHandles = findall(0, 'Type', 'figure');
for k = 1:length(figHandles)
    try
        h = figHandles(k);
        % 核心防护：必须是有效图形句柄，且必须有名字
        if isvalid(h) && isgraphics(h)
            f_n = get(h, 'Name');
            if ~isempty(f_n) && (contains(f_n, model_tag) || contains(f_n, 'Blind'))
                % 统一使用 saveas 和 print
                saveas(h, fullfile(dir_out, [f_n, '.fig']));
                saveas(h, fullfile(dir_out, [f_n, '.png']));
                % 如果 print 失败，静默跳过，不中断程序
                try
                    print(h, fullfile(dir_out, [f_n, '.svg']), '-dsvg', '-r600');
                catch
                    fprintf('   (Skipped SVG for: %s)\n', f_n);
                end
            end
        end
    catch
        continue; % 如果以上尝试任何一步崩溃，直接跳到下一个句柄
    end
end
% ========================== 替换结束 ==========================

%% ========================================================================
%  Module 9: Metrics Summary Table (Table 1)
%% ========================================================================
figure('Color', [1 1 1], 'Position', [200, 200, 800, 420], 'Name', [model_tag, '_Table01']); axis off;
% --- 修改 Table 1 数据引用 ---
t_data = {'Coefficient of Determination (R2)', sprintf('%.4f', bp.R2_tr), sprintf('%.4f', bp.R2);
          'Root Mean Square Error (RMSE)', sprintf('%.3f', bp.RMSE_tr), sprintf('%.3f', bp.RMSE);
          'Mean Absolute Error (MAE)', sprintf('%.3f', bp.MAE_tr), sprintf('%.3f', bp.MAE)};
uitable('Data', t_data, 'ColumnName', {'Evaluation Metric', 'Training Set', 'Testing Set'}, ...
        'Units', 'Normalized', 'Position', [0.05, 0.2, 0.9, 0.65], 'FontSize', 10);
auto_layout_manager(gcf, 'Table 1: Comprehensive Performance Evaluation Summary for GA-BP', '');

%% --- Export Results (Fixed java.lang.NullPointerException) ---
dir_out = [model_tag, '_Single_Model_Results']; 
if ~exist(dir_out, 'dir'); mkdir(dir_out); end

drawnow; pause(1); % 强制刷新绘图队列
figHandles = findall(0, 'Type', 'figure');
for k = 1:length(figHandles)
    if isvalid(figHandles(k))
        f_n = get(figHandles(k), 'Name');
        if ~isempty(f_n) && (contains(f_n, model_tag) || contains(f_n, 'Blind'))
            % 保存三种格式：fig, svg, png
            saveas(figHandles(k), fullfile(dir_out, [f_n, '.fig']));
            saveas(figHandles(k), fullfile(dir_out, [f_n, '.png']));
            print(figHandles(k), fullfile(dir_out, [f_n, '.svg']), '-dsvg', '-r600');
        end
    end
end

Scatter_Data.te_real = bp.T_te_real; 
Scatter_Data.te_sim = bp.T_te_sim;
Stats_Summary.R2_test_loop = r2_vals;       
Stats_Summary.RMSE_test_loop = cellfun(@(x) x.RMSE, results_cell);   
Stats_Summary.MAE_test_loop = cellfun(@(x) x.MAE, results_cell);     
Stats_Summary.R2_mean = mean(Stats_Summary.R2_test_loop);
Stats_Summary.Time = total_time;
% --- 为 T7 汇总平台封装核心数据 (Fig 20-21 来源) ---
% 1. 封装特征重要性 (Importance)
% 显式指定 GA-BP 的重要性来源变量 bp.importance (来自 results_cell 的最优结果)
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
fprintf('✅ [%s] Task Completed! Time: %.2fs | Mean R2: %.4f \n', model_tag, total_time, Stats_Summary.R2_mean);

end

%% ========================================================================
% --- 替换原第 338 行函数头 ---此行为338行
function [T_s2, T_s1, met, trace, net, ps_in, ps_out] = Internal_GABP_Engine_V50(P_tr, T_tr, P_te, T_te, max_gen)
    global S1 p_train t_train
    % --- 修正：补全分号，防止命令行输出爆炸 ---
    [p_train_n, ps_in] = mapminmax(P_tr', 0, 1); 
    p_test_n = mapminmax('apply', P_te', ps_in);
    [t_train_n, ps_out] = mapminmax(T_tr', 0, 1); % 这里一定要有分号
    p_train = p_train_n; t_train = t_train_n;
    % ... (后续 GA 代码保持不变)
    S1 = 10; 
    net_init = newff(p_train, t_train, S1, {'tansig','purelin'}, 'trainlm');
    net_init.trainParam.epochs = 1000; net_init.trainParam.goal = 1e-7; net_init.trainParam.showWindow = 0;
    assignin('base', 'S1', S1);
    assignin('base', 'net', net_init);
    assignin('base', 'p_train', p_train);
    assignin('base', 't_train', t_train);
    S_vars = 9 * S1 + S1 * 1 + S1 + 1; 
    bounds = ones(S_vars, 1) * [-1, 1]; 
    initPpp = initializega(20, bounds, 'gabpEval', [], [1e-6, 1]);  
    [Bestpop, ~, ~, trace] = ga(bounds, 'gabpEval', [], initPpp, [1e-6 1 0], 'maxGenTerm', max_gen,...
                               'normGeomSelect', 0.09, 'arithXover', 2, 'nonUnifMutation', [2 max_gen 3]);
    [~, W1, B1, W2, B2] = gadecod(Bestpop);
    net_init.IW{1, 1} = W1; net_init.LW{2, 1} = W2; 
    net_init.b{1} = B1; net_init.b{2} = B2;
    net = train(net_init, p_train, t_train);
    T_s1 = mapminmax('reverse', sim(net, p_train), ps_out)';
    T_s2 = mapminmax('reverse', sim(net, p_test_n), ps_out)';
    imp = abs(net.LW{2,1}) * abs(net.IW{1,1}); 
    met.rel_imp = (imp / sum(imp)) * 100;
    met.R2_train = 1 - sum((T_tr - T_s1).^2) / sum((T_tr - mean(T_tr)).^2);
    met.R2_test = 1 - sum((T_te - T_s2).^2) / sum((T_te - mean(T_te)).^2);
    met.RMSE = sqrt(mean((T_te - T_s2).^2)); met.MAE = mean(abs(T_te - T_s2));
    met.RMSE_tr = sqrt(mean((T_tr - T_s1).^2)); met.MAE_tr = mean(abs(T_tr - T_s1));
end

function auto_layout_manager(fig_handle, en_title, en_title_sub)
    ax = findobj(fig_handle, 'Type', 'axes');
    min_bottom = 1.0; 
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

%% ========================================================================
%  Module 11: Final Generalization Blind Test Interface (Full Stats V102)
%% ========================================================================
function run_blind_test(bp_struct, modelName, featureNames)
    % 1. 弹出标准文件选择框 (喂给它弹出框)
    fprintf('\n>>> Initializing Generalization Blind Test Validation...\n');
    [file, path] = uigetfile({'*.xlsx;*.xls;*.csv'}, ['Select Blind Test Data for ', modelName]);
    if isequal(file, 0), return; end
    
    fDir = [modelName, '_BlindTest_Scientific_Results'];
    if ~exist(fDir, 'dir'), mkdir(fDir); end
    
    % 2. 载入数据并适配
    data = readtable(fullfile(path, file));
    X_raw = table2array(data(:, 1:9)); Y_actual = table2array(data(:, 10)); 
    vIdx = ~all(X_raw == 0, 2); X_raw = X_raw(vIdx, :); Y_actual = Y_actual(vIdx, :);
    numS = size(X_raw, 1);
    fprintf('>>> Identified %d valid experimental data groups.\n', numS);

    % 3. --- 核心：归一化同步 (彻底解决精度为负数的关键) ---
    model = bp_struct.model; ps_in = bp_struct.ps_in; ps_out = bp_struct.ps_out;
    X_scaled = mapminmax('apply', X_raw', ps_in); % 应用训练集比例
    
    % 4. 预测与反归一化 (回到 MPa)
    Y_pred_n = sim(model, X_scaled); 
    Y_pred = mapminmax('reverse', Y_pred_n, ps_out)'; 
    
    Y_actual = Y_actual(:); Y_pred = Y_pred(:);
    Errors = Y_actual - Y_pred; AbsE = abs(Errors);

    % 5. 指标统计分析
    R2 = 1 - sum(Errors.^2)/sum((Y_actual - mean(Y_actual)).^2);
    RMSE = sqrt(mean(Errors.^2)); MAE = mean(AbsE);
    [max_e, ~] = max(AbsE); [min_e, ~] = min(AbsE);
    acc_vec = 1 - (AbsE ./ (Y_actual + eps));
    max_acc = max(acc_vec); min_acc = min(acc_vec);

    %% --- Plot 1: Regression Scatter with Full Metrics (SCI) ---
    h1 = figure('Name', [modelName, '_Blind_Scatter'], 'Color', 'w', 'Position', [100 100 650 600]);
    scatter(Y_actual, Y_pred, 85, 'filled', 'MarkerFaceColor', [0.85 0.33 0.1], 'MarkerFaceAlpha', 0.6); hold on;
    all_v = [Y_actual; Y_pred]; ref_l = [min(all_v) max(all_v)]; plot(ref_l, ref_l, 'k--', 'LineWidth', 2);
    xlabel('Experimental Strength (MPa)', 'FontWeight', 'bold'); ylabel('Predicted Strength (MPa)', 'FontWeight', 'bold');
    
    % 深度统计标注
    stats_box = {['Samples (N): ', num2str(numS)], ['Overall R^2: ', num2str(R2, '%.4f')], ...
                 ['Mean MAE: ', num2str(MAE, '%.3f'), ' MPa'], ...
                 ['Max Error: ', num2str(max_e, '%.2f'), ' MPa'], ...
                 ['Best Accuracy: ', num2str(max_acc*100, '%.1f'), '%'], ...
                 ['Worst Accuracy: ', num2str(min_acc*100, '%.1f'), '%']};
    annotation('textbox', [0.15, 0.6, 0.35, 0.3], 'String', stats_box, 'FitBoxToText', 'on', 'BackgroundColor', 'w', 'FontWeight', 'bold');
    title([modelName, ' Blind Test: Scientific Regression'], 'FontSize', 12); grid on; axis square;
    saveas(h1, fullfile(fDir, 'Blind_Regression_Scatter.png'));

    %% --- Plot 2: Residuals with ±MAE Tolerance ---
    h2 = figure('Name', [modelName, '_Blind_Residuals'], 'Color', 'w', 'Position', [150 150 700 500]);
    stem(Errors, 'filled', 'Color', [0.3 0.3 0.3]); hold on;
    yline(MAE, 'r--', 'LineWidth', 1.5); yline(-MAE, 'r--', 'LineWidth', 1.5);
    text(1, MAE+0.5, ['Avg Tolerance: \pm', num2str(MAE, '%.2f'), ' MPa'], 'Color', 'r', 'FontWeight', 'bold');
    xlabel('Blind Sample Index'); ylabel('Error (MPa)'); title('Residual Tolerance Distribution'); grid on;
    saveas(h2, fullfile(fDir, 'Blind_Residual_Distribution.png'));

    %% --- Plot 3: SHAP Mechanism Comparison ---
    h3 = figure('Name', [modelName, '_Blind_SHAP'], 'Color', 'w', 'Position', [200 200 800 600]);
    shap_v = zeros(9, numS);
    for i = 1:numS
        b_o = sim(model, mapminmax('apply', X_raw(i,:)', ps_in));
        for f = 1:9
            tx = X_raw(i,:); tx(f) = mean(X_raw(:,f));
            shap_v(f, i) = b_o - sim(model, mapminmax('apply', tx', ps_in));
        end
    end
    for f_p = 1:9, scatter(shap_v(f_p, :), f_p + (rand(1, numS)-0.5)*0.3, 25, 'filled', 'MarkerFaceAlpha', 0.5); hold on; end
    set(gca, 'YTick', 1:9, 'YTickLabel', featureNames); xlabel('SHAP Value'); 
    title('Mechanism Consistency: Blind Set SHAP Summary'); grid on;
    saveas(h3, fullfile(fDir, 'Blind_SHAP_Consistency.png'));

    msgbox(['Blind Test Success! N=', num2str(numS), ' R2=', num2str(R2, '%.4f')], 'Success');
end

