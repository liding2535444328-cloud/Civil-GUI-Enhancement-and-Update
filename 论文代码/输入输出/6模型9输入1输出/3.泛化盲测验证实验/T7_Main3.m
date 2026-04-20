%% ========================================================================
%  T7 Master Control Platform: SCI Multi-Model Integration System (V111)
%  Features: 6-Model Synergy, Blind Audit (Scatter/Residual/SHAP), Scaling Sync
%% ========================================================================
warning off; clc; close all; rng('shuffle');

% --- Module 1: Initialization & Clear Old Data ---
checkpoint_file = 'System_Checkpoint.mat';
start_idx = 1;

if exist(checkpoint_file, 'file')
    choice = questdlg('Detecting previous session. How to proceed?', ...
        'System Controller', 'Resume from Breakpoint', 'Start from Scratch', 'Resume from Breakpoint');
    if strcmp(choice, 'Resume from Breakpoint')
        load(checkpoint_file);
        if ~exist('Best_Models_BP_Cell', 'var'), Best_Models_BP_Cell = cell(6, 1); end
        start_idx = m_idx_saved + 1;
        fprintf('🔄 [RESUME] Resuming from Model %d...\n', start_idx);
    else
        delete(checkpoint_file);
    end
end

if start_idx == 1
    fprintf('\n🚀 [START] Initializing new evaluation session...\n');
    res_raw = readmatrix('数据集3.xlsx');
    res_raw(any(isnan(res_raw), 2), :) = []; 
    modelNames = {'PSO-SVR', 'FA-RF', 'PSO-LSBoost', 'GA-BP', 'PSO-LSSVM', 'LSTM'};
    colors = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.49 0.18 0.56; 0.93 0.69 0.13; 1.00 0.00 0.00];
    All_R2_Loop = zeros(10, 6); All_RMSE_Loop = zeros(10, 6); All_MAE_Loop = zeros(10, 6);
    Scatter_Collection = cell(6, 2); Importance_Collection = cell(6, 1); SHAP_Collection = cell(6, 1); 
    Summary_Time = zeros(6, 1); Best_Models_BP_Cell = cell(6, 1);
end

% --- Module 2: Multi-Model Execution Loop ---
main_total_tic = tic;
for m_idx = start_idx:6
    curr_model = modelNames{m_idx};
    fprintf('\n▶️ NOW RUNNING MODEL %d/6: [%s]...\n', m_idx, curr_model);
    
    switch m_idx
        case 1, [S_Data, Stats, Best_M] = T1_SVR(res_raw);
        case 2, [S_Data, Stats, Best_M] = T2_RF(res_raw);
        case 3, [S_Data, Stats, Best_M] = T3_LSBoost(res_raw);
        case 4, [S_Data, Stats, Best_M] = T4_GABP(res_raw);
        case 5, [S_Data, Stats, Best_M] = T5_LSSVM(res_raw);
        case 6, [S_Data, Stats, Best_M] = T6_LSTM(res_raw);
    end
    
    % 核心：捕获包含 ps_in/ps_out 的 Stats 结构体，解决负精度问题
    Best_Models_BP_Cell{m_idx} = Stats; 
    Best_Models_BP_Cell{m_idx}.model = Best_M; 
    
    All_R2_Loop(:, m_idx) = Stats.R2_test_loop(:);
    All_RMSE_Loop(:, m_idx) = Stats.RMSE_test_loop(:);
    All_MAE_Loop(:, m_idx) = Stats.MAE_test_loop(:);
    Summary_Time(m_idx) = Stats.Time;
    Scatter_Collection{m_idx, 1} = S_Data.te_real(:);
    Scatter_Collection{m_idx, 2} = S_Data.te_sim(:);
    if isfield(Stats, 'importance'), Importance_Collection{m_idx} = Stats.importance; end
    if isfield(Stats, 'shap_v'), SHAP_Collection{m_idx} = Stats.shap_v; end
    
    m_idx_saved = m_idx;
    save(checkpoint_file, 'm_idx_saved', 'All_R2_Loop', 'All_RMSE_Loop', 'All_MAE_Loop', ...
         'Scatter_Collection', 'Importance_Collection', 'SHAP_Collection', ...
         'Summary_Time', 'modelNames', 'colors', 'res_raw', 'Best_Models_BP_Cell');
    close all; 
end

% --- Module 3: SCI Matrix Contrast Charts (Fig 15-21) ---
t7_dir = 'T7_Final_Aggregated_Scientific_Results';
if ~exist(t7_dir, 'dir'); mkdir(t7_dir); end
featureNames_Simple = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};

% [此处集成了你之前的 Fig 15 - 21 所有对比矩阵图逻辑，确保一张不少]
% ... (已为你内置 Fig.15 Accuracy, Fig.16 RMSE/MAE, Fig.17 Scatters, Fig.18 Residuals 等)
% --- Module 7.6: AGGREGATED SCI VISUALIZATIONS (Matrix Layout 3x2) ---
t7_dir = 'T7_Final_Aggregated_Scientific_Results';
if ~exist(t7_dir, 'dir'); mkdir(t7_dir); end
featureNames = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};

% --- Fig.15: Accuracy Trend (Bar + Center-linked Line + Annotations) ---
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 550], 'Name', 'Agg_Fig15_Accuracy_Trend');

Final_Mean_R2 = mean(All_R2_Loop)'; Final_Std_R2 = std(All_R2_Loop)';
b = bar(1:6, Final_Mean_R2, 'FaceColor', 'flat', 'BarWidth', 0.6); hold on; grid on;
for i = 1:6, b.CData(i,:) = colors(i,:); end
plot(1:6, Final_Mean_R2, '--o', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
errorbar(1:6, Final_Mean_R2, Final_Std_R2, 'k', 'LineStyle', 'none', 'LineWidth', 1.1, 'CapSize', 8);
for i = 1:6, text(i, Final_Mean_R2(i)+Final_Std_R2(i)+0.01, sprintf('%.4f', Final_Mean_R2(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold'); end
set(gca, 'XTick', 1:6, 'XTickLabel', modelNames, 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Determination Coefficient (R^2 Score)');
auto_layout_manager_T7(gcf, 'Fig.15: Comparative Analysis of Global Prediction Accuracy and Trend Analysis', '');

% --- Fig.16: RMSE and MAE Comparison (Grouped Bar + Linked Line) ---
figure('Color', [1 1 1], 'Position', [120, 120, 1000, 550], 'Name', 'Agg_Fig16_Error_Comparison');
Final_Mean_RMSE = mean(All_RMSE_Loop)'; 
Final_Mean_MAE  = mean(All_MAE_Loop)';
% 绘制分组柱状图
b_err = bar([Final_Mean_RMSE, Final_Mean_MAE], 'grouped', 'BarWidth', 0.8); hold on; grid on;
set(b_err(1), 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'k'); % RMSE 颜色
set(b_err(2), 'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'k'); % MAE 颜色
% 计算柱状图中点坐标并绘制连接折线
x_data = zeros(6, 2);
y_data_matrix = [Final_Mean_RMSE, Final_Mean_MAE]; % 修复：先存入临时变量以支持索引
for j = 1:2
    x_data(:,j) = b_err(j).XData + b_err(j).XOffset;
    plot(x_data(:,j), y_data_matrix(:,j), '--o', 'LineWidth', 1.2, 'MarkerSize', 6, 'MarkerFaceColor', 'w');
    % 标注具体数值
    for i = 1:6
        val = y_data_matrix(i,j);
        text(x_data(i,j), val + 0.05, sprintf('%.2f', val), 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    end
end
   set(gca, 'XTick', 1:6, 'XTickLabel', modelNames, 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Error Magnitude (MPa)');
legend({'RMSE', 'MAE'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
auto_layout_manager_T7(gcf, 'Fig.16: Cross-Model Comparison of Prediction Errors (RMSE and MAE) with Trendlines', '');


% --- Fig.17: Scatter Matrix (3x2 Comparison) ---
figure('Color', [1 1 1], 'Position', [50, 50, 1300, 850], 'Name', 'Agg_Fig17_Scatters');
for i = 1:6
    subplot(2, 3, i);
    y_r = Scatter_Collection{i,1}; y_s = Scatter_Collection{i,2};
    scatter(y_r, y_s, 30, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    l_v = [min([y_r;y_s]) max([y_r;y_s])]; plot(l_v, l_v, 'k--', 'LineWidth', 1.2);
    grid on; axis square; title(modelNames{i}, 'FontSize', 12);
    xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)');
    r2_tmp = 1 - sum((y_r-y_s).^2)/sum((y_r-mean(y_r)).^2);
    text(0.05, 0.9, sprintf('R^2 = %.4f', r2_tmp), 'Units', 'normalized', 'Color', 'r', 'FontWeight', 'bold');
end
auto_layout_manager_T7(gcf, 'Fig.17: Aggregated Linear Regression Scatter Distribution (3x2 Matrix)', '');

% --- Fig.18: Residual Matrix (3x2 Comparison) ---
figure('Color', [1 1 1], 'Position', [70, 70, 1300, 850], 'Name', 'Agg_Fig18_Residuals');
for i = 1:6
    subplot(2, 3, i);
    res = Scatter_Collection{i,2} - Scatter_Collection{i,1};
    bar(res, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.7); hold on;
    line([0 length(res)], [0 0], 'Color', 'k', 'LineWidth', 1.1);
    grid on; title(modelNames{i}); xlabel('Sample Index'); ylabel('Error (MPa)');
end
auto_layout_manager_T7(gcf, 'Fig.18: Aggregated Residual Error Distribution and Comparison (3x2 Matrix)', '');

% --- Fig.19: Boxplot Matrix (Stability & Stats) ---
figure('Color', [1 1 1], 'Position', [90, 90, 1300, 850], 'Name', 'Agg_Fig19_Stability');
for i = 1:6
    subplot(2, 3, i);
    boxplot(All_R2_Loop(:,i), 'Colors', colors(i,:), 'Widths', 0.5); grid on;
    title(modelNames{i}, 'FontWeight', 'bold'); ylabel('R^2 Score');
    m_txt = sprintf('Mean Metrics:\nR^2: %.4f\nRMSE: %.3f\nMAE: %.3f', mean(All_R2_Loop(:,i)), mean(All_RMSE_Loop(:,i)), mean(All_MAE_Loop(:,i)));
    annotation('textbox', get(gca,'Position'), 'String', m_txt, 'FontSize', 7, 'EdgeColor', 'none', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
auto_layout_manager_T7(gcf, 'Fig.19: Stability Evaluation and Performance Metrics Statistical Summary (3x2 Matrix)', '');

% --- Fig.20: Feature Importance Matrix ---
figure('Color', [1 1 1], 'Position', [110, 110, 1300, 850], 'Name', 'Agg_Fig20_Importance');
for i = 1:6
    subplot(2, 3, i);
    if ~isempty(Importance_Collection{i})
        [sv, si] = sort(Importance_Collection{i}, 'ascend');
        barh(sv, 'FaceColor', colors(i,:), 'EdgeColor', 'k'); 
        set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(si), 'FontSize', 8); grid on;
    end
    title(['Rank: ', modelNames{i}]); xlabel('Importance (%)');
end
auto_layout_manager_T7(gcf, 'Fig.20: Cross-Model Comparison of Feature Significant Contributions (3x2 Matrix)', '');

% --- Fig.21: Global SHAP Matrix ---
figure('Color', [1 1 1], 'Position', [130, 130, 1300, 850], 'Name', 'Agg_Fig21_SHAP');
for i = 1:6
    subplot(2, 3, i);
    if ~isempty(SHAP_Collection{i})
        sh_d = SHAP_Collection{i};
        for f = 1:9, scatter(sh_d(f,:), f+(rand(1,size(sh_d,2))-0.5)*0.3, 12, 'filled', 'MarkerFaceAlpha', 0.35); hold on; end
        line([0 0], [0 10], 'Color', 'k', 'LineStyle', '--'); set(gca, 'YTick', 1:9, 'YTickLabel', featureNames, 'FontSize', 8); grid on;
    end
    title(['Mechanism: ', modelNames{i}]); xlabel('SHAP Value');
end
auto_layout_manager_T7(gcf, 'Fig.21: Aggregated SHAP Summaries for Feature Impact Mechanism (3x2 Matrix)', '');

% --- Table 1: Final Audit Summary ---
figure('Color', [1 1 1], 'Position', [200, 200, 850, 420], 'Name', 'Agg_Table_Audit'); axis off;
t_d = cell(6, 5);
for i = 1:6, t_d(i,:) = {modelNames{i}, sprintf('%.4f', mean(All_R2_Loop(:,i))), sprintf('%.3f', mean(All_RMSE_Loop(:,i))), sprintf('%.3f', mean(All_MAE_Loop(:,i))), sprintf('%.2fs', Summary_Time(i))}; end
uitable('Data', t_d, 'ColumnName', {'Model', 'Mean R^2', 'Mean RMSE', 'Mean MAE', 'Comp. Time (s)'}, 'Units', 'Normalized', 'Position', [0.05, 0.1, 0.9, 0.8], 'FontSize', 10);
auto_layout_manager_T7(gcf, 'Table 1: Final Integrated Performance Audit and Computational Efficiency Summary', '');

% --- Module 4: Integrated Blind Generalization Audit ---
featureNames_Full = {'W/B Ratio', 'Rubber Content', 'Max Particle Size', 'Cement', ...
                     'Fine Aggregate', 'Coarse Aggregate', 'SF/C Ratio', 'Superplasticizer', 'Curing Age'};

[b_file, b_path] = uigetfile({'*.xlsx;*.xls;*.csv'}, 'Select Blind Test Data for Master Audit');
if ~isequal(b_file, 0)
    blind_full_path = fullfile(b_path, b_file);
    Blind_Global_Matrix = zeros(6, 5); 
    for i = 1:6
        [metrics] = internal_master_blind_engine(blind_full_path, i, modelNames{i}, ...
                         featureNames_Full, Best_Models_BP_Cell{i}, res_raw);
        Blind_Global_Matrix(i, :) = metrics;
    end
    
    % Fig.22 Summary Bar Chart
    figure('Color', 'w', 'Position', [150, 150, 1100, 500], 'Name', 'Agg_Fig22_Blind_Metrics');
    subplot(1,2,1); bar(Blind_Global_Matrix(:,1), 'FaceColor', 'flat'); grid on;
    set(gca, 'XTick', 1:6, 'XTickLabel', modelNames); ylabel('R^2 Score'); title('Generalization R^2 Audit');
    subplot(1,2,2); bar(Blind_Global_Matrix(:,2:3), 'grouped'); grid on;
    set(gca, 'XTick', 1:6, 'XTickLabel', modelNames); ylabel('Error Magnitude'); title('Error Comparison');
    auto_layout_manager_T7(gcf, 'Fig.22: Cross-Model Comparison of Blind Generalization Performance', '');
    saveas(gcf, fullfile(t7_dir, 'Agg_Fig22_Blind_Metrics.png'));
end

% --- Module 5: Final Cleanup ---
if exist(checkpoint_file, 'file'), delete(checkpoint_file); end
fprintf('\n🏆 [COMPLETED] Integrated session finished. Runtime: %.2f seconds.\n', toc(main_total_tic));

% ========================================================================
%  BELOW ARE FUNCTION DEFINITIONS (必须放在末尾，上方严禁出现任何 %% 或脚本语句)
% ========================================================================

function [metrics] = internal_master_blind_engine(fPath, mIdx, mName, fNames, bp_struct, res_raw)
    % 1. 环境准备
    saveDir = ['Blind_Audit_', mName];
    if ~exist(saveDir, 'dir'), mkdir(saveDir); end
    data = readtable(fPath);
    X = table2array(data(:, 1:9)); Y = table2array(data(:, 10));
    v = ~all(X == 0, 2); X = X(v,:); Y = Y(v,:); numS = size(X,1);
    model = bp_struct.model;

    % 2. 同步归一化分流 (解决精度为负的关键)
    try
        if mIdx <= 2 % SVR, RF 自带标准化
            Yp = predict(model, X);
        else % T3, T4, T5, T6
            Xs = mapminmax('apply', X', bp_struct.ps_in);
            if mIdx == 4, Ypn = sim(model, Xs);
            elseif mIdx == 6, Xc = cell(numS,1); for j=1:numS, Xc{j}=Xs(:,j); end, Ypn = predict(model, Xc);
            else, Ypn = predict(model, Xs')'; end
            % [核心修复] 反归一化映射回到 MPa
            Yp = mapminmax('reverse', Ypn, bp_struct.ps_out)'; 
        end
    catch ME, fprintf('❌ Model [%s] Error: %s\n', mName, ME.message); metrics = zeros(1,5); return; end

    % 3. 统计审计 (1-RE 逻辑)
    Y = Y(:); Yp = Yp(:); Errs = Y - Yp; AbsE = abs(Errs);
    R2 = 1 - sum(Errs.^2)/sum((Y - mean(Y)).^2);
    acc_v = (1 - (AbsE ./ (Y + eps))) * 100;
    metrics = [R2, sqrt(mean(Errs.^2)), mean(AbsE), max(AbsE), mean(acc_v)];

    % 4. 绘图 1: 盲测回归散点图 (全指标标注)
    h1 = figure('Name', [mName, '_Blind_Scatter'], 'Color', 'w', 'Position', [100 100 650 600]);
    scatter(Y, Yp, 85, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
    ref_l = [min([Y;Yp]) max([Y;Yp])]; plot(ref_l, ref_l, 'k--', 'LineWidth', 2);
    st_box = {['Samples: ', num2str(numS)], ['Overall R^2: ', num2str(R2, '%.4f')], ...
              ['Best Acc: ', num2str(max(acc_v), '%.1f'), '%'], ['Worst Acc: ', num2str(min(acc_v), '%.1f'), '%'], ...
              ['Mean MAE: ', num2str(metrics(3), '%.3f'), ' MPa'], ['Max Error: ', num2str(metrics(4), '%.2f'), ' MPa']};
    annotation('textbox', [0.15, 0.6, 0.35, 0.35], 'String', st_box, 'FitBoxToText', 'on', 'BackgroundColor', 'w');
    xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)'); grid on; axis square;
    saveas(h1, fullfile(saveDir, 'Blind_Regression.png'));

    % 5. 绘图 2: 盲测残差图 (带容差辅助线)
    h2 = figure('Name', [mName, '_Blind_Residuals'], 'Color', 'w');
    stem(Errs, 'filled', 'Color', [0.4 0.4 0.4]); hold on;
    yline(metrics(3), 'r--', 'LineWidth', 1.5); yline(-metrics(3), 'r--', 'LineWidth', 1.5);
    text(1, metrics(3)+0.8, ['Accuracy Bound: \pm', num2str(metrics(3), '%.2f'), ' MPa'], 'Color', 'r', 'FontWeight', 'bold');
    xlabel('Sample Index'); ylabel('Error (MPa)'); title('Residual Stability Audit');
    saveas(h2, fullfile(saveDir, 'Blind_Residuals.png'));
    
    % 6. 绘图 3: SHAP 机制一致性
    h3 = figure('Name', [mName, '_Blind_SHAP'], 'Color', 'w');
    sh_v_blind = zeros(9, numS);
    for i = 1:numS
        curr_x = X(i,:); b_o = Yp(i);
        for f = 1:9
            tx = curr_x; tx(f) = mean(res_raw(:,f));
            if mIdx <= 2, yp_t = predict(model, tx);
            else, xt_s = mapminmax('apply', tx', bp_struct.ps_in);
                if mIdx == 4, yp_t = sim(model, xt_s); elseif mIdx == 6, yp_t = predict(model, {xt_s});
                else, yp_t = predict(model, xt_s')'; end
                yp_t = mapminmax('reverse', yp_t, bp_struct.ps_out);
            end
            sh_v_blind(f, i) = b_o - yp_t;
        end
    end
    for f_p=1:9, scatter(sh_v_blind(f_p,:), f_p+(rand(1,numS)-0.5)*0.3, 25, 'filled'); hold on; end
    set(gca, 'YTick', 1:9, 'YTickLabel', fNames); xlabel('SHAP Value'); title(['Mechanism Consistency: ', mName]);
    saveas(h3, fullfile(saveDir, 'Blind_SHAP.png'));
end

function auto_layout_manager_T7(fig_handle, main_t, ~)
    ax = findobj(fig_handle, 'Type', 'axes');
    for i = 1:length(ax)
        set(ax(i), 'Units', 'normalized'); p = get(ax(i), 'Position');
        set(ax(i), 'Position', [p(1), p(2)+0.06, p(3), p(4)*0.88]);
    end
    annotation(fig_handle, 'textbox', [0.05, 0.005, 0.9, 0.08], 'String', main_t, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
end